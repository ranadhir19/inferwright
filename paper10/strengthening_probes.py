"""Coordinator strengthening probes for Paper 10 (cycle 2).
Fixes tester items: representative trace sampling, cap-bias accounting,
prefill+decode residence model, consistent product-vs-joint comparison,
waste-definition check on the low-load negative case.
Writes strengthening_results.json."""
import sys, os, json, csv
import numpy as np
sys.path.insert(0, os.path.expanduser("~/workspace/llm-inference-engine/src"))
from llm_inference_engine.paged_kv.blocks import BlockManager, ContiguousAllocator
from llm_inference_engine.paged_kv.workload import Request, gen_workload
from llm_inference_engine.spec_decode import expected_speedup, optimal_k

W = os.path.dirname(os.path.abspath(__file__))
BUDGET = 1_000_000; MAX_LEN = 2048
out = {}

def simulate(make_allocator, requests):
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
            else: rej += 1
        else:
            if r.seq_id in admitted: alloc.free(r.seq_id); admitted.discard(r.seq_id)
        maxc = max(maxc, alloc.live_sequences())
    return {"waste_pct_agg": 100.0 * waste / reserved if reserved else 0.0,
            "max_concurrent": maxc, "admitted": adm, "rejected": rej}

def arrivals_uniform(n, seed=11, window=600.0):
    rng = np.random.default_rng(seed); return np.sort(rng.uniform(0, window, size=n))

# ---- trace population accounting ----
path = os.path.join(W, "data", "BurstGPT_1.csv")
allrows = []; posrows = []
with open(path) as f:
    for row in csv.DictReader(f):
        try: req = int(row["Request tokens"]); resp = int(row["Response tokens"]); ts = float(row["Timestamp"])
        except Exception: continue
        if req > 0: allrows.append((ts, req, resp))
        if resp > 0 and req > 0: posrows.append((ts, req, resp))
capped = [r for r in posrows if r[1] + r[2] <= MAX_LEN]
out["population"] = {"rows_req_pos": len(allrows), "rows_resp_pos": len(posrows),
                     "rows_within_cap": len(capped),
                     "cap_drop_pct_of_resp_pos": 100.0 * (len(posrows) - len(capped)) / len(posrows),
                     "mean_total_within_cap": float(np.mean([r[1] + r[2] for r in capped]))}
posrows.sort(); capped.sort()

# systematic sample: every k-th capped row spanning the whole trace, 2000 rows
step = len(capped) // 2000
sample = capped[::step][:2000]
prompts = np.array([r[1] for r in sample]); resps = np.array([r[2] for r in sample]); totals = prompts + resps
out["systematic_sample"] = {"step": int(step), "n": len(sample), "mean_prompt": float(prompts.mean()),
                            "mean_response": float(resps.mean()), "mean_total": float(totals.mean()),
                            "median_total": float(np.median(totals))}
first2000 = capped[:2000]
out["first2000_mean_total"] = float(np.mean([r[1] + r[2] for r in first2000]))

def make_reqs(totals_, prompts_, resps_, rho=32.0, S=1.0, seed=11):
    """Residence = prompt/rho (prefill) + response/S (decode, spec-scaled). Allocation length = total."""
    arr = arrivals_uniform(len(totals_), seed=seed)
    residence = prompts_ / rho + resps_ / S
    return [Request(seq_id=i, arrival=float(arr[i]), length=int(totals_[i]),
                    departure=float(arr[i] + residence[i])) for i in range(len(totals_))]

# ---- main trace run on systematic sample, corrected residence (rho=32), spec off ----
main = {}
rq = make_reqs(totals, prompts, resps, rho=32.0, S=1.0)
for B in [8, 16, 32, 64]:
    main[f"B{B}"] = simulate(lambda B=B: BlockManager(BUDGET, B), rq)
main["contiguous"] = simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), rq)
def mean_waste_paged(L, B):
    a = np.ceil(L / B) * B; return float(100 * ((a - L) / a).mean())
main["mean_def"] = {f"B{B}": mean_waste_paged(totals, B) for B in [8, 16, 32, 64]}
main["mean_def"]["contiguous"] = float(100 * ((MAX_LEN - totals) / MAX_LEN).mean())
pred = {}
for B in [8, 16, 32, 64]:
    a = np.ceil(totals / B) * B; pred[f"B{B}"] = float(100 * (a - totals).sum() / a.sum())
main["pred_waste_agg_formula"] = pred
out["main_systematic_rho32"] = main

# residence sensitivity for spec-off paged16/contig admissions
sens_rho = {}
for rho in [8.0, 32.0, 128.0, 1e9]:
    rqx = make_reqs(totals, prompts, resps, rho=rho, S=1.0)
    sens_rho[f"rho={rho:g}"] = {"paged16": simulate(lambda: BlockManager(BUDGET, 16), rqx),
                                "contiguous": simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), rqx)}
out["residence_sensitivity_specoff"] = sens_rho

# ---- joint sim on systematic sample, corrected residence ----
joint = {}
for label, alpha, c, forced_k in [("spec_off", None, None, None), ("a0.7_c0.1", 0.7, 0.1, None),
                                  ("a0.85_c0.1", 0.85, 0.1, None), ("a0.3_c0.1_opt", 0.3, 0.1, None),
                                  ("a0.3_k8_forced", 0.3, 0.1, 8)]:
    if alpha is None: S = 1.0; k = 0
    elif forced_k: k = forced_k; S = expected_speedup(k, alpha, c)
    else: k, S = optimal_k(alpha, c, k_max=8)
    rqx = make_reqs(totals, prompts, resps, rho=32.0, S=S)
    joint[label] = {"k": int(k), "S": float(S),
                    "paged16": simulate(lambda: BlockManager(BUDGET, 16), rqx),
                    "contiguous": simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), rqx)}
# consistent product: (trace spec-off admission ratio paged/contig) x S, vs joint admission gain (paged+spec)/(contig spec-off)
base_ratio = joint["spec_off"]["paged16"]["admitted"] / joint["spec_off"]["contiguous"]["admitted"]
prod = {}
for label, v in joint.items():
    if label == "spec_off": continue
    prod[label] = {"product_admission_gain": base_ratio * v["S"],
                   "joint_admission_gain": v["paged16"]["admitted"] / joint["spec_off"]["contiguous"]["admitted"],
                   "product_error_pct": 100.0 * (base_ratio * v["S"] - v["paged16"]["admitted"] / joint["spec_off"]["contiguous"]["admitted"]) / (v["paged16"]["admitted"] / joint["spec_off"]["contiguous"]["admitted"])}
out["joint_systematic_rho32"] = joint
out["product_vs_joint_consistent"] = {"trace_specoff_admission_ratio": base_ratio, "rows": prod}

# ---- low-load waste definition check (tester item: 1.99% label) ----
wl = gen_workload(np.random.default_rng(11), n=200, arrival_window=600.0, max_len=2048)
agg = simulate(lambda: BlockManager(BUDGET, 16), wl)
Ls = np.array([r.length for r in wl])
out["lowload_check"] = {"aggregate_pct": agg["waste_pct_agg"],
                        "mean_def_pct_recomputed": mean_waste_paged(Ls, 16),
                        "note": "builder analysis.py reported aggregate 1.987% for this workload"}

# ---- alpha_min boundary restatement data: S<1 share on restricted grids ----
def neg_share(alo, ahi):
    res = {}
    for c in [0.05, 0.1, 0.2]:
        cells = [(a, k) for a in np.arange(alo, ahi + 1e-9, 0.01) for k in range(1, 9)]
        res[str(c)] = float(np.mean([expected_speedup(k, a, c) < 1 for a, k in cells]))
    return res
out["neg_share_full_grid"] = neg_share(0.01, 0.99)
out["neg_share_alpha_0.30_0.95"] = neg_share(0.30, 0.95)
out["neg_share_alpha_ge_0.50"] = neg_share(0.50, 0.99)

with open(os.path.join(W, "strengthening_results.json"), "w") as f: json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
print("WROTE strengthening_results.json")
