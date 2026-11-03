"""Coordinator audit probes for Paper 10 (independent of builder's analysis.py).
A: recompute headline trace numbers + waste-definition reconciliation.
B: TRUE joint simulation (spec speed scales service time inside the allocator sim).
C: trace-slice and budget sensitivity.
Writes audit_results.json."""
import sys, os, json, csv
import numpy as np
sys.path.insert(0, os.path.expanduser("~/workspace/llm-inference-engine/src"))
from llm_inference_engine.paged_kv.blocks import BlockManager, ContiguousAllocator
from llm_inference_engine.paged_kv.workload import Request
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

# load trace
path = os.path.join(W, "data", "BurstGPT_1.csv")
rows = []
with open(path) as f:
    for row in csv.DictReader(f):
        try: req = int(row["Request tokens"]); resp = int(row["Response tokens"]); ts = float(row["Timestamp"])
        except Exception: continue
        if resp > 0 and req + resp <= MAX_LEN and req > 0: rows.append((ts, req, resp))
rows.sort()
out["n_trace_valid_recomputed"] = len(rows)

def arrivals_uniform(n, seed=11, window=600.0):
    rng = np.random.default_rng(seed); return np.sort(rng.uniform(0, window, size=n))

def make_requests(lengths, speed=1.0, seed=11):
    arr = arrivals_uniform(len(lengths), seed=seed)
    return [Request(seq_id=i, arrival=float(arr[i]), length=int(lengths[i]),
                    departure=float(arr[i] + lengths[i] / speed)) for i in range(len(lengths))]

# ---- A: headline slice 0:2000 + waste definitions ----
sel = rows[:2000]; lengths = np.array([r[1] + r[2] for r in sel])
reqs = make_requests(lengths)
a = {}
for B in [8, 16, 32, 64]:
    a[f"B{B}"] = simulate(lambda B=B: BlockManager(BUDGET, B), reqs)
a["contiguous"] = simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), reqs)
def mean_waste_paged(lengths, B):
    allocd = np.ceil(lengths / B) * B; return float(100 * ((allocd - lengths) / allocd).mean())
def mean_waste_contig(lengths):
    return float(100 * ((MAX_LEN - lengths) / MAX_LEN).mean())
a["mean_waste_pct_per_admission_def"] = {f"B{B}": mean_waste_paged(lengths, B) for B in [8, 16, 32, 64]}
a["mean_waste_pct_per_admission_def"]["contiguous"] = mean_waste_contig(lengths)
out["A_headline_slice0"] = a

# ---- B: true joint simulation on trace lengths ----
# spec ON: service time / S(k*, alpha, c). Compare admitted/concurrency/waste for paged16 vs contig, spec on/off.
joint = {}
for label, alpha, c in [("spec_off", None, None), ("a0.7_c0.1", 0.7, 0.1), ("a0.85_c0.1", 0.85, 0.1), ("a0.3_c0.1", 0.3, 0.1), ("a0.3_k8_forced", 0.3, 0.1)]:
    if alpha is None: S = 1.0; k = 0
    elif label == "a0.3_k8_forced":
        from llm_inference_engine.spec_decode import expected_speedup as es; S = es(8, alpha, c); k = 8
    else: k, S = optimal_k(alpha, c, k_max=8)
    rq = make_requests(lengths, speed=S)
    joint[label] = {"k": int(k), "S": float(S),
                    "paged16": simulate(lambda: BlockManager(BUDGET, 16), rq),
                    "contiguous": simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), rq)}
out["B_joint_simulation"] = joint

# ---- C: sensitivity: second slice, random slice, budgets ----
sens = {}
sel2 = rows[2000:4000]; lengths2 = np.array([r[1] + r[2] for r in sel2])
rq2 = make_requests(lengths2)
sens["slice2000_4000"] = {"paged16": simulate(lambda: BlockManager(BUDGET, 16), rq2),
                          "contiguous": simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), rq2),
                          "mean_total_len": float(lengths2.mean())}
rng = np.random.default_rng(3); idx = rng.choice(len(rows), 2000, replace=False)
lengths3 = np.array([rows[i][1] + rows[i][2] for i in sorted(idx)])
rq3 = make_requests(lengths3)
sens["random2000_seed3"] = {"paged16": simulate(lambda: BlockManager(BUDGET, 16), rq3),
                            "contiguous": simulate(lambda: ContiguousAllocator(BUDGET, MAX_LEN), rq3),
                            "mean_total_len": float(lengths3.mean())}
for bud in [500_000, 2_000_000]:
    sens[f"budget_{bud}"] = {"paged16": simulate(lambda bud=bud: BlockManager(bud, 16), reqs),
                             "contiguous": simulate(lambda bud=bud: ContiguousAllocator(bud, MAX_LEN), reqs)}
out["C_sensitivity"] = sens

with open(os.path.join(W, "audit_results.json"), "w") as f: json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
print("WROTE audit_results.json")
