"""Hardware-anchored speculative-decoding point (CPU).
Target: openai-community/gpt2 (124M). Draft: distilbert/distilgpt2 (82M).
Greedy speculative decoding, manual KV-cache implementation.
Invariant: at cycle start both caches cover prefix P and each model's
next-token argmax after P is known. A cycle drafts k tokens with the
draft model, verifies them in ONE target block forward, emits the
accepted prefix plus one correction/bonus token. The correction token
is materialised at the next cycle start by one single-token forward of
each model (counted in the target-forward tally, disclosed in the paper).
Losslessness: spec output must equal vanilla greedy output exactly.
Prompts: excerpts of Project Gutenberg #1342 (Pride and Prejudice).
Outputs hw_results.json."""
import json, time, os, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(2)
W=os.path.dirname(os.path.abspath(__file__))
K=4; NGEN=48; NPROMPT=6; PLEN=32

url="https://www.gutenberg.org/files/1342/1342-0.txt"
txtpath=os.path.join(W,"data","gutenberg1342.txt")
if not os.path.exists(txtpath):
    urllib.request.urlretrieve(url, txtpath)
text=open(txtpath,encoding="utf-8",errors="ignore").read()
start=text.find("It is a truth universally acknowledged")
body=text[start:]
chunks=[body[i*4000:(i+1)*4000] for i in range(NPROMPT)]

tok=AutoTokenizer.from_pretrained("openai-community/gpt2")
target=AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
draft=AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2").eval()
prompts=[]
for c in chunks:
    ids=tok(c,return_tensors="pt").input_ids[0,:PLEN]
    prompts.append(ids.unsqueeze(0))

def slice_cache(past, length):
    if past is None: return None
    if hasattr(past, "layers"):  # transformers Cache API: slice layer tensors
        for layer in past.layers:
            if getattr(layer, "is_initialized", False) and layer.keys is not None:
                layer.keys = layer.keys[:, :, :length, :]
                layer.values = layer.values[:, :, :length, :]
        return past
    return tuple(tuple(t[:,:,:length,:] for t in layer) for layer in past)

@torch.no_grad()
def vanilla(ids, n):
    out=target.generate(ids, max_new_tokens=n, do_sample=False,
                        pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:].tolist()

@torch.no_grad()
def spec(ids, n, k=K):
    o=target(ids); t_past=o.past_key_values; t_next=int(torch.argmax(o.logits[0,-1]))
    o=draft(ids);  d_past=o.past_key_values; d_next=int(torch.argmax(o.logits[0,-1]))
    L=ids.shape[1]; output=[]; pending=None
    proposed=0; accepted=0; t_forwards=1; d_forwards=1
    pos_prop=[0]*k; pos_acc=[0]*k; cycles=0; emitted_total=0
    while len(output)<n:
        if pending is not None:
            feed=torch.tensor([[pending]])
            o=target(feed, past_key_values=t_past); t_past=o.past_key_values; t_forwards+=1
            t_next=int(torch.argmax(o.logits[0,-1]))
            o=draft(feed, past_key_values=d_past); d_past=o.past_key_values; d_forwards+=1
            d_next=int(torch.argmax(o.logits[0,-1]))
            L+=1; pending=None
        # draft k proposals; pr[0]=d_next
        pr=[d_next]; dp=d_past
        for _ in range(k-1):
            o=draft(torch.tensor([[pr[-1]]]), past_key_values=dp); dp=o.past_key_values; d_forwards+=1
            pr.append(int(torch.argmax(o.logits[0,-1])))
        proposed+=k; cycles+=1
        for i in range(k): pos_prop[i]+=1
        # target verification: one block forward of the k proposals
        o=target(torch.tensor([pr]), past_key_values=t_past); t_forwards+=1
        t_block_past=o.past_key_values
        cand=[t_next]+[int(torch.argmax(o.logits[0,i])) for i in range(k-1)]
        bonus=int(torch.argmax(o.logits[0,-1]))
        a=0
        while a<k and pr[a]==cand[a]: a+=1
        accepted+=a
        for i in range(a): pos_acc[i]+=1
        emitted=pr[:a]+[cand[a] if a<k else bonus]; emitted_total+=len(emitted)
        if a==k:  # draft cache lacks the last proposal (predicted, never fed)
            o=draft(torch.tensor([[pr[-1]]]), past_key_values=dp); dp=o.past_key_values; d_forwards+=1
        # caches keep only P + accepted prefix
        t_past=slice_cache(t_block_past, L+a)
        d_past=slice_cache(dp, L+a)
        L=L+a
        output.extend(emitted); pending=emitted[-1]
        # t_next/d_next for the new P will be set when pending is materialised;
        # but if loop ends here they are stale -- loop guard handles it.
        if len(output)>=n: break
        # set placeholder; real values computed at cycle start after pending feed.
        # draft's next after P: need draft logits at end of P -- the drafting pass
        # computed logits after each proposal; after truncation the relevant next
        # prediction is recomputed at next cycle start via the pending feed for
        # the target, and for the draft we recompute below by a fresh look:
        # (the pending feed at cycle start updates d_next from the fed token.)
    return output[:n], {"proposed":proposed,"accepted":accepted,
                        "t_forwards":t_forwards,"d_forwards":d_forwards,
                        "pos_prop":pos_prop,"pos_acc":pos_acc,
                        "cycles":cycles,"emitted_total":emitted_total}

# NOTE: after a cycle, d_next must be the draft prediction after the new P.
# In the loop above, when pending is fed at the next cycle start, d_next is
# refreshed from that feed (prediction after P+pending). But drafting in that
# cycle should start from P+pending, which is exactly what the refreshed
# d_next gives. The stale d_next is never used before refresh because
# pending is not None at every cycle start after the first. Good.

def timeit(fn, *a):
    t0=time.perf_counter(); r=fn(*a); return r, time.perf_counter()-t0

# single-token forward latency (cached, context 80) for c-hat
@torch.no_grad()
def forward_latency(model, reps=60):
    ids=prompts[0].repeat(1,3)[:, :80]
    o=model(ids); past=o.past_key_values; feed=ids[:, -1:]
    ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); o=model(feed, past_key_values=past); ts.append(time.perf_counter()-t0)
        past=o.past_key_values
    ts.sort(); return ts[len(ts)//2]

lat_t=forward_latency(target); lat_d=forward_latency(draft)
chat=lat_d/lat_t
print(f"single-token latency: target {lat_t*1000:.2f} ms, draft {lat_d*1000:.2f} ms, c-hat {chat:.4f}")

res={"k":K,"ngen":NGEN,"nprompt":NPROMPT,"plen":PLEN,
     "lat_target_ms":lat_t*1000,"lat_draft_ms":lat_d*1000,"c_hat":chat,
     "prompts":[]}
tot_v_tok=0; tot_v_time=0.0; tot_s_tok=0; tot_s_time=0.0
tot_prop=0; tot_acc=0; tot_tf=0
for i,p in enumerate(prompts):
    v,vt=timeit(vanilla,p,NGEN)
    t0=time.perf_counter(); s_out,meta=spec(p,NGEN); st=time.perf_counter()-t0
    match = (s_out==v)
    res["prompts"].append({"i":i,"vanilla_time_s":vt,"spec_time_s":st,
        "vanilla_tps":NGEN/vt,"spec_tps":NGEN/st,"match":match,**meta})
    print(f"prompt {i}: vanilla {NGEN/vt:.2f} tok/s, spec {NGEN/st:.2f} tok/s, match={match}, acc={meta['accepted']}/{meta['proposed']}, t_fw={meta['t_forwards']}")
    tot_v_tok+=NGEN; tot_v_time+=vt; tot_s_tok+=NGEN; tot_s_time+=st
    tot_prop+=meta["proposed"]; tot_acc+=meta["accepted"]; tot_tf+=meta["t_forwards"]
alpha_hat=tot_acc/tot_prop if tot_prop else 0
tot_cycles=sum(p["cycles"] for p in res["prompts"])
tot_emitted=sum(p["emitted_total"] for p in res["prompts"])
pos_prop_agg=[sum(p["pos_prop"][i] for p in res["prompts"]) for i in range(K)]
pos_acc_agg=[sum(p["pos_acc"][i] for p in res["prompts"]) for i in range(K)]
emp_tokens_per_cycle=tot_emitted/tot_cycles
# effective alpha: solve sum_{i=0}^k a^i = emp_tokens_per_cycle
lo,hi=0.0,1.0
for _ in range(80):
    mid=(lo+hi)/2
    if sum(mid**i for i in range(K+1))<emp_tokens_per_cycle: lo=mid
    else: hi=mid
alpha_eff=(lo+hi)/2
v_tps=tot_v_tok/tot_v_time; s_tps=tot_s_tok/tot_s_time
E=(1-alpha_hat**(K+1))/(1-alpha_hat) if alpha_hat<1 else K+1
pred=E/(1+K*chat)
res.update({"alpha_hat":alpha_hat,"tokens_per_target_forward":tot_s_tok/tot_tf,
  "cycles":tot_cycles,"emitted_total_incl_overshoot":tot_emitted,
  "emp_tokens_per_cycle":emp_tokens_per_cycle,"alpha_eff":alpha_eff,
  "pos_prop":pos_prop_agg,"pos_acc":pos_acc_agg,
  "pos_accept_rate":[pos_acc_agg[i]/pos_prop_agg[i] for i in range(K)],
  "predicted_speedup_alpha_eff":emp_tokens_per_cycle/(1+K*chat),
  "vanilla_tps_agg":v_tps,"spec_tps_agg":s_tps,"measured_speedup":s_tps/v_tps,
  "E_tokens_formula":E,"predicted_speedup":pred,
  "deviation_measured_minus_predicted":s_tps/v_tps-pred,
  "all_match":all(p["match"] for p in res["prompts"])})
print(json.dumps({k2:res[k2] for k2 in ["alpha_hat","tokens_per_target_forward","vanilla_tps_agg","spec_tps_agg","measured_speedup","predicted_speedup","all_match"]},indent=1))
with open(os.path.join(W,"hw_results.json"),"w") as f: json.dump(res,f,indent=1)
print("WROTE hw_results.json")
