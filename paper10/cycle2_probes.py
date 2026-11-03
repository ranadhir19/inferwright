"""Paper 10 — Elevation Cycle 2 probes (C2-1, C2-2, C2-3).

Frozen protocol (see Cycle2_Log_2026-11-03.md for the logged resolutions):
- Step 0 validates this harness by reproducing scaled_joint_results.json
  exactly from strengthening_probes.py's machinery before any variation.
- C2-1: trace-timestamp arrivals (affinely rescaled to the 600-step window),
  same systematic sample, same oracle allocators. Answers objection 3.
- C2-2: incremental (non-oracle) paged allocator, same sample, same uniform
  arrivals; preemption on growth failure; gains on completed (primary)
  and admitted (secondary). Answers objection 4.
- C2-3: second systematic offset (308) + seeded random sample (seed 23),
  headline protocol unchanged; spread of the four product errors.
Writes cycle2_results.json. $0, CPU only.
"""
import sys, os, json, csv, math
import numpy as np
sys.path.insert(0, os.path.expanduser("~/workspace/llm-inference-engine/src"))
from llm_inference_engine.paged_kv.blocks import BlockManager, ContiguousAllocator
from llm_inference_engine.paged_kv.workload import Request
from llm_inference_engine.spec_decode import expected_speedup, optimal_k

W = os.path.dirname(os.path.abspath(__file__))
PW = os.path.dirname(W)  # paper10_work/
MAX_LEN = 2048
RHO = 32.0
BUDGET = 250_000
out = {"protocol": "cycle2 frozen; see Cycle2_Log_2026-11-03.md"}

# ---------------------------------------------------------------- population
path = os.path.join(PW, "data", "BurstGPT_1.csv")
allrows, posrows = [], []
with open(path) as f:
    for row in csv.DictReader(f):
        try:
            req = int(row["Request tokens"]); resp = int(row["Response tokens"]); ts = float(row["Timestamp"])
        except Exception:
            continue
        if req > 0:
            allrows.append((ts, req, resp))
        if resp > 0 and req > 0:
            posrows.append((ts, req, resp))
posrows.sort()
capped = [r for r in posrows if r[1] + r[2] <= MAX_LEN]
capped.sort()
pop = {"rows_req_pos": len(allrows), "rows_resp_pos": len(posrows),
       "rows_within_cap": len(capped),
       "cap_drop_pct_of_resp_pos": 100.0 * (len(posrows) - len(capped)) / len(posrows),
       "mean_total_within_cap": float(np.mean([r[1] + r[2] for r in capped]))}
ref = json.load(open(os.path.join(PW, "strengthening_results.json")))
rp = ref["population"]
assert pop["rows_req_pos"] == rp["rows_req_pos"], (pop, rp)
assert pop["rows_resp_pos"] == rp["rows_resp_pos"]
assert pop["rows_within_cap"] == rp["rows_within_cap"]
assert abs(pop["mean_total_within_cap"] - rp["mean_total_within_cap"]) < 1e-9
out["population_check"] = {"restored_trace_matches_strengthening_record": True, **pop}
print("population check OK:", pop)

step = len(capped) // 2000
assert step == 617, step

def sample_stats(rows):
    p = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows]); t = p + s
    return {"n": len(rows), "mean_prompt": float(p.mean()), "mean_response": float(s.mean()),
            "mean_total": float(t.mean()), "median_total": float(np.median(t))}

sample = capped[::step][:2000]
ss = sample_stats(sample)
rs = ref["systematic_sample"]
assert ss["n"] == 2000 and abs(ss["mean_prompt"] - rs["mean_prompt"]) < 1e-9
assert abs(ss["mean_response"] - rs["mean_response"]) < 1e-9 and abs(ss["mean_total"] - rs["mean_total"]) < 1e-9
out["systematic_sample_check"] = {"step": step, **ss}
print("systematic sample check OK:", ss)

# ---------------------------------------------------------------- machinery
def simulate(make_allocator, requests):
    """Verbatim from strengthening_probes.py (the shipped event simulator)."""
    alloc = make_allocator(); events = []
    for r in requests:
        events.append((r.arrival, 1, "arrive", r)); events.append((r.departure, 0, "depart", r))
    events.sort(key=lambda e: (e[0], e[1]))
    admitted = set(); adm = 0; rej = 0; waste = 0; reserved = 0; maxc = 0
    for t, _, kind, r in events:
        if kind == "arrive":
            a0 = alloc.allocated_slots()
            if alloc.alloc(r.seq_id, r.length) is not None:
                admitted.add(r.seq_id); adm += 1; res = alloc.allocated_slots() - a0
                waste += res - r.length; reserved += res
            else:
                rej += 1
        else:
            if r.seq_id in admitted:
                alloc.free(r.seq_id); admitted.discard(r.seq_id)
        maxc = max(maxc, alloc.live_sequences())
    return {"waste_pct_agg": 100.0 * waste / reserved if reserved else 0.0,
            "max_concurrent": maxc, "admitted": adm, "rejected": rej}

def arrivals_uniform(n, seed=11, window=600.0):
    rng = np.random.default_rng(seed); return np.sort(rng.uniform(0, window, size=n))

def make_reqs(totals_, prompts_, resps_, arrivals, S=1.0):
    residence = prompts_ / RHO + resps_ / S
    return [Request(seq_id=i, arrival=float(arrivals[i]), length=int(totals_[i]),
                    departure=float(arrivals[i] + residence[i])) for i in range(len(totals_))]

CONDITIONS = [("spec_off", None, None, None), ("a0.7_c0.1", 0.7, 0.1, None),
              ("a0.85_c0.1", 0.85, 0.1, None), ("a0.3_c0.1_opt", 0.3, 0.1, None),
              ("a0.3_k8_forced", 0.3, 0.1, 8)]

def condition_ks():
    res = {}
    for label, alpha, c, forced_k in CONDITIONS:
        if alpha is None:
            res[label] = (0, 1.0)
        elif forced_k:
            res[label] = (forced_k, float(expected_speedup(forced_k, alpha, c)))
        else:
            k, S = optimal_k(alpha, c, k_max=8)
            res[label] = (int(k), float(S))
    return res

KS = condition_ks()
sj = json.load(open(os.path.join(PW, "scaled_joint_results.json")))
for label, (k, S) in KS.items():
    refc = sj["joint_250k"][label]
    assert refc["k"] == k and abs(refc["S"] - S) < 1e-12, (label, k, S, refc)
print("condition (k,S) check OK:", KS)

def joint_run(rows, arrivals):
    prompts = np.array([r[1] for r in rows], dtype=float)
    resps = np.array([r[2] for r in rows], dtype=float)
    totals = prompts + resps
    joint = {}
    for label, (k, S) in KS.items():
        rq = make_reqs(totals, prompts, resps, arrivals, S=S)
        joint[label] = {"k": k, "S": S,
                        "paged16": simulate(lambda: BlockManager(BUDGET, 16), rq),
                        "contiguous": simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), rq)}
    return joint

def gains(joint, paged_key="admitted", contig_key="admitted"):
    base = joint["spec_off"]["paged16"][paged_key] / joint["spec_off"]["contiguous"][contig_key]
    denom = joint["spec_off"]["contiguous"][contig_key]
    rows = {}
    for label, v in joint.items():
        if label == "spec_off":
            continue
        jg = v["paged16"][paged_key] / denom
        prod = base * v["S"]
        rows[label] = {"joint_gain": jg, "product": prod,
                       "product_error_pct": 100.0 * (prod - jg) / jg}
    return {"base_ratio": base, "rows": rows}

# ---------------------------------------------------------------- STEP 0: harness validation
arr_u = arrivals_uniform(2000)
prompts_s = np.array([r[1] for r in sample], dtype=float)
resps_s = np.array([r[2] for r in sample], dtype=float)
totals_s = prompts_s + resps_s
scan = {}
for B_ in [100_000, 250_000, 500_000]:
    rq = make_reqs(totals_s, prompts_s, resps_s, arr_u, S=1.0)
    scan[f"budget_{B_}"] = {"paged16": simulate(lambda B_=B_: BlockManager(B_, 16), rq),
                            "contiguous": simulate(lambda B_=B_: ContiguousAllocator(B_, MAX_LEN), rq)}
joint0 = joint_run(sample, arr_u)
ok = True
for bk, bv in sj["budget_scan"].items():
    for alloc in ["paged16", "contiguous"]:
        for field in ["admitted", "rejected", "max_concurrent"]:
            if scan[bk][alloc][field] != bv[alloc][field]:
                ok = False; print("MISMATCH scan", bk, alloc, field, scan[bk][alloc][field], bv[alloc][field])
        if abs(scan[bk][alloc]["waste_pct_agg"] - bv[alloc]["waste_pct_agg"]) > 1e-9:
            ok = False; print("MISMATCH scan waste", bk, alloc)
for label, v in sj["joint_250k"].items():
    for alloc in ["paged16", "contiguous"]:
        for field in ["admitted", "rejected", "max_concurrent"]:
            if joint0[label][alloc][field] != v[alloc][field]:
                ok = False; print("MISMATCH joint", label, alloc, field, joint0[label][alloc][field], v[alloc][field])
        if abs(joint0[label][alloc]["waste_pct_agg"] - v[alloc]["waste_pct_agg"]) > 1e-9:
            ok = False; print("MISMATCH joint waste", label, alloc)
assert abs(sj["base_ratio"] - gains(joint0)["base_ratio"]) < 1e-12
assert ok, "STEP 0 FAILED — harness does not reproduce scaled_joint_results.json"
out["step0_harness_validation"] = {"reproduces_scaled_joint_results_json": True,
                                   "budget_scan": scan, "joint_250k": joint0,
                                   "gains": gains(joint0)}
print("STEP 0 OK — scaled_joint_results.json reproduced exactly")

# ---------------------------------------------------------------- C2-1: trace-timestamp arrivals
ts = np.array([r[0] for r in sample], dtype=float)
scale = 600.0 / (ts[-1] - ts[0])
arr_trace = (ts - ts[0]) * scale
gaps_t = np.diff(arr_trace)
gaps_u = np.diff(arr_u)
diag = {"rescale_factor": float(scale), "trace_span_seconds_raw": float(ts[-1] - ts[0]),
        "interarrival_cv_trace_rescaled": float(gaps_t.std() / gaps_t.mean()),
        "interarrival_cv_uniform_protocol": float(gaps_u.std() / gaps_u.mean()),
        "max_gap_trace": float(gaps_t.max()), "max_gap_uniform": float(gaps_u.max()),
        "zero_gaps_trace": int((gaps_t == 0).sum()),
        "mean_residence_specoff": float((prompts_s / RHO + resps_s).mean())}
joint1 = joint_run(sample, arr_trace)
out["c2_1_trace_arrivals"] = {"diagnostics": diag, "joint_250k": joint1, "gains": gains(joint1)}
print("C2-1 done:", json.dumps(gains(joint1), indent=1))

# ---------------------------------------------------------------- C2-2: incremental paged allocator
def simulate_incremental(prompts_, resps_, arrivals, S=1.0, budget=BUDGET, B=16, admit_full=False):
    """Paged allocation as real systems do it: prompt blocks at admission,
    one block claimed at each decode block-boundary; growth failure preempts
    (blocks freed, work lost). admit_full=True is the oracle self-check mode.
    Event priority at equal times: depart(0) < grow(1) < arrive(2)."""
    nblocks = budget // B
    free = nblocks
    n = len(prompts_)
    held = np.zeros(n, dtype=int)
    status = np.zeros(n, dtype=int)  # 0 pending, 1 live, 2 rejected, 3 preempted, 4 done
    events = []
    for i in range(n):
        P, R = int(prompts_[i]), int(resps_[i])
        a = float(arrivals[i])
        residence = P / RHO + R / S
        events.append((a, 2, i))
        if not admit_full:
            held0 = math.ceil(P / B)
            final = math.ceil((P + R) / B)
            for m in range(held0, final):
                j = m * B + 1 - P  # token index at which block m+1 is first needed
                if 1 <= j <= R:
                    events.append((a + P / RHO + j / S, 1, i))
        events.append((a + residence, 0, i))
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    adm = comp = pre = rej = 0; live = 0; maxc = 0; minfree = nblocks
    for t, prio, i in events:
        if prio == 2:  # arrive
            need = math.ceil((int(prompts_[i]) + (int(resps_[i]) if admit_full else 0)) / B)
            if free >= need:
                free -= need; held[i] = need; status[i] = 1; adm += 1; live += 1
            else:
                status[i] = 2; rej += 1
        elif prio == 1:  # grow by one block
            if status[i] != 1:
                continue
            if free >= 1:
                free -= 1; held[i] += 1
            else:
                free += held[i]; held[i] = 0; status[i] = 3; pre += 1; live -= 1
        else:  # depart
            if status[i] == 1:
                free += held[i]; held[i] = 0; status[i] = 4; comp += 1; live -= 1
        maxc = max(maxc, live); minfree = min(minfree, free)
    return {"admitted": adm, "completed": comp, "preempted": pre, "rejected": rej,
            "max_concurrent": maxc, "peak_slots_held": (nblocks - minfree) * B}

# oracle self-check: admit_full mode must reproduce BlockManager spec-off admissions
chk = simulate_incremental(prompts_s, resps_s, arr_u, S=1.0, admit_full=True)
assert chk["admitted"] == 1844 and chk["completed"] == 1844 and chk["preempted"] == 0, chk
print("C2-2 oracle self-check OK:", chk)

joint2 = {}
for label, (k, S) in KS.items():
    joint2[label] = {"k": k, "S": S,
                     "paged16_incremental": simulate_incremental(prompts_s, resps_s, arr_u, S=S),
                     "contiguous": joint0[label]["contiguous"]}
joint2g = {l: {"paged16": v["paged16_incremental"], "contiguous": v["contiguous"], "S": v["S"]}
           for l, v in joint2.items()}
out["c2_2_incremental_allocator"] = {
    "joint_250k": joint2,
    "gains_completed": gains(joint2g, paged_key="completed"),
    "gains_admitted": gains(joint2g, paged_key="admitted")}
print("C2-2 done (completed):", json.dumps(out["c2_2_incremental_allocator"]["gains_completed"], indent=1))
print("C2-2 done (admitted):", json.dumps(out["c2_2_incremental_allocator"]["gains_admitted"], indent=1))

# ---------------------------------------------------------------- C2-3: second samples
offset_rows = capped[step // 2::step][:2000]
rng = np.random.default_rng(23)
ridx = np.sort(rng.choice(len(capped), size=2000, replace=False))
random_rows = [capped[i] for i in ridx]
c23 = {"offset_systematic": {"offset": step // 2, "stats": sample_stats(offset_rows),
                             "joint_250k": joint_run(offset_rows, arr_u)},
       "random_seed23": {"stats": sample_stats(random_rows),
                         "joint_250k": joint_run(random_rows, arr_u)}}
for name in c23:
    c23[name]["gains"] = gains(c23[name]["joint_250k"])
out["c2_3_second_samples"] = c23
spread = {}
g0 = gains(joint0)["rows"]
for label in g0:
    errs = [g0[label]["product_error_pct"], c23["offset_systematic"]["gains"]["rows"][label]["product_error_pct"],
            c23["random_seed23"]["gains"]["rows"][label]["product_error_pct"]]
    spread[label] = {"errors_pct_three_samples": errs, "min": min(errs), "max": max(errs),
                     "range": max(errs) - min(errs)}
out["c2_3_spread"] = {"base_ratios": [gains(joint0)["base_ratio"], c23["offset_systematic"]["gains"]["base_ratio"],
                                      c23["random_seed23"]["gains"]["base_ratio"]],
                      "per_condition": spread}
print("C2-3 spread:", json.dumps(spread, indent=1))

with open(os.path.join(W, "cycle2_results.json"), "w") as f:
    json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("WROTE cycle2_results.json")
