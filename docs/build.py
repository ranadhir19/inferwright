#!/usr/bin/env python3
"""Builds the LLM Inference Engine landing site into docs/ (GitHub Pages ready).

Each project page: Problem / The gap / The solution / Architecture (SVG) /
Results / Insights / Implementation. Problem/Gap/Solution/Insights are parsed
from each project's README.md at build time, so the site stays in sync.

Run: python3 docs/build.py
"""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CSS = """
:root{
  --bg:#0a0e13; --panel:#101722; --panel2:#0d141e; --border:#223349;
  --text:#e9edf4; --muted:#99a4ba; --faint:#6b7690;
  --accent:#5eead4; --green:#7ee2a8; --amber:#f0b35e; --red:#f08787; --blue:#6ea8fe;
  --radius:14px;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  line-height:1.65;font-size:16px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px}
nav.top{position:sticky;top:0;z-index:50;background:rgba(10,14,19,.92);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
nav.top .wrap{display:flex;align-items:center;gap:28px;height:60px}
.brand{font-weight:700;font-size:17px;color:var(--text);letter-spacing:.2px}
.brand span{color:var(--accent)}
nav.top .links{margin-left:auto;display:flex;gap:20px;font-size:14px}
nav.top .links a{color:var(--muted)}
nav.top .links a:hover{color:var(--text)}
.hero{padding:84px 0 60px;text-align:left;
  background:radial-gradient(900px 380px at 20% 0%,rgba(94,234,212,.10),transparent 60%)}
.hero .kicker{color:var(--accent);font-size:13px;font-weight:700;letter-spacing:2.5px;
  text-transform:uppercase;margin-bottom:18px}
.hero h1{font-size:46px;line-height:1.12;margin:0 0 20px;letter-spacing:-.5px;max-width:820px}
.hero p.sub{color:var(--muted);font-size:18px;max-width:720px;margin:0 0 34px}
.btn{display:inline-block;background:var(--accent);color:#0a0e13;font-weight:700;
  padding:12px 26px;border-radius:10px;font-size:15px}
.btn:hover{text-decoration:none;filter:brightness(1.08)}
.btn.ghost{background:transparent;color:var(--text);border:1px solid var(--border);margin-left:12px}
.install{margin-top:34px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.install code{background:#0d141e;border:1px solid var(--border);border-radius:8px;
  padding:10px 18px;font-size:15px;color:var(--green)}
.install span{color:var(--muted);font-size:14px}
.stats{display:flex;gap:36px;margin-top:44px;flex-wrap:wrap}
.stats div b{font-size:26px;display:block}
.stats div span{color:var(--muted);font-size:13px}
section{padding:56px 0}
h2.sec{font-size:28px;margin:0 0 8px;letter-spacing:-.3px}
p.lead{color:var(--muted);max-width:760px;margin:0 0 30px}
.cat{border:1px solid var(--border);border-radius:var(--radius);padding:28px;
  margin-bottom:28px;background:var(--panel)}
.cat h3{margin:0 0 4px;font-size:21px}
.cat h3 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:10px}
.cat p.desc{color:var(--muted);margin:0 0 20px;max-width:720px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:760px){.cards{grid-template-columns:1fr}.hero h1{font-size:34px}}
.card{background:var(--panel2);border:1px solid var(--border);border-radius:12px;
  padding:22px;display:flex;flex-direction:column}
.card h4{margin:0 0 6px;font-size:18px}
.card h4 a{color:var(--text)}
.card .tag{font-size:12px;color:var(--faint);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px}
.card p{color:var(--muted);font-size:14.5px;margin:0 0 16px;flex:1}
.card .metric{font-size:14px;color:var(--text);border-top:1px solid var(--border);padding-top:12px}
.card .metric b{color:var(--green)}
.proj-hero{padding:56px 0 30px}
.proj-hero .kicker{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}
.proj-hero h1{font-size:38px;margin:0 0 10px;letter-spacing:-.4px}
.proj-hero p.tagline{color:var(--muted);font-size:18px;margin:0 0 8px}
.crumb{font-size:13px;color:var(--faint);margin-bottom:22px}
.crumb a{color:var(--faint)}
.block{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);
  padding:28px;margin-bottom:22px}
.block h2{font-size:20px;margin:0 0 12px}
.block h2 .n{display:inline-block;width:28px;height:28px;border-radius:8px;background:var(--accent);
  color:#0a0e13;font-size:15px;text-align:center;line-height:28px;margin-right:10px;font-weight:800}
.block p{margin:0 0 12px;color:#c9d1e2}
.block p:last-child{margin-bottom:0}
.block ul,.block ol{margin:8px 0 0;padding-left:20px;color:#c9d1e2}
.block li{margin-bottom:8px}
.block code{background:#0d141e;border:1px solid var(--border);border-radius:5px;
  padding:1px 6px;font-size:13.5px;color:var(--accent)}
.diagram{background:#0c1219;border:1px solid var(--border);border-radius:12px;
  padding:18px;overflow-x:auto;margin:6px 0 4px}
.diagram svg{display:block;margin:0 auto;max-width:100%;height:auto}
.caption{font-size:13px;color:var(--faint);text-align:center;margin-top:8px}
table.res{width:100%;border-collapse:collapse;font-size:14px;margin:10px 0}
table.res th,table.res td{border:1px solid var(--border);padding:9px 12px;text-align:left}
table.res th{background:var(--panel2);color:var(--muted);font-weight:600}
.hl{color:var(--green);font-weight:700}
img.chart{width:100%;border-radius:10px;border:1px solid var(--border);margin-top:10px}
code.tree{background:#0c1219;border:1px solid var(--border);border-radius:10px;
  display:block;padding:16px 18px;font-size:13.5px;color:#c9d1e2;overflow-x:auto;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre}
.pagenav{display:flex;justify-content:space-between;margin:40px 0 60px;gap:16px}
.pagenav a{border:1px solid var(--border);border-radius:10px;padding:14px 20px;
  color:var(--text);font-size:14px;background:var(--panel)}
.pagenav a small{display:block;color:var(--faint);font-size:12px}
footer{border-top:1px solid var(--border);padding:34px 0 48px;color:var(--faint);font-size:13.5px}
footer .wrap{display:flex;justify-content:space-between;gap:20px;flex-wrap:wrap}
"""

SVG_DEFS = """
<defs>
  <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#8fa2c8"/>
  </marker>
  <marker id="arra" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#5eead4"/>
  </marker>
  <marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#7ee2a8"/>
  </marker>
</defs>
"""

def box(x, y, w, h, label, sub="", fill="#131b28", stroke="#3b4d75", tcolor="#e9edf4", fs=13):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    s += f'<text x="{x+w/2}" y="{y+h/2-2}" text-anchor="middle" fill="{tcolor}" font-size="{fs}" font-weight="600">{label}</text>'
    if sub:
        s += f'<text x="{x+w/2}" y="{y+h/2+15}" text-anchor="middle" fill="#99a4ba" font-size="11">{sub}</text>'
    return s

def arrow(x1, y1, x2, y2, color="#8fa2c8", marker="arr", dashed=False, label=""):
    d = ' stroke-dasharray="5,4"' if dashed else ""
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6"{d} marker-end="url(#{marker})"/>'
    if label:
        s += f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2-7}" text-anchor="middle" fill="{color}" font-size="11">{label}</text>'
    return s

def svg_wrap(inner, w, h):
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{inner}</svg>'

# ---------------------------------------------------------------------------
# Architecture diagrams
# ---------------------------------------------------------------------------

def svg_transformer():
    b = [SVG_DEFS]
    b.append(box(20, 150, 110, 54, "Tokens", "embedding"))
    # decoder block x N
    b.append(box(180, 60, 300, 250, "", "", fill="#0f1622", stroke="#3b4d75"))
    b.append('<text x="330" y="86" text-anchor="middle" fill="#e9edf4" font-size="13" font-weight="700">Decoder block x N</text>')
    b.append(box(205, 105, 250, 44, "RMSNorm", "pre-norm", fs=12))
    b.append(box(205, 160, 250, 44, "GQA Attention", "RoPE · KV cache", fs=12, fill="#1a2b26", stroke="#3f7a68"))
    b.append(box(205, 215, 250, 44, "RMSNorm + SwiGLU FFN", "gated MLP", fs=12))
    b.append('<text x="330" y="292" text-anchor="middle" fill="#99a4ba" font-size="11">residuals around both sublayers</text>')
    b.append(arrow(130, 177, 178, 177))
    b.append(arrow(480, 177, 528, 177))
    b.append(box(530, 150, 110, 54, "Logits", "lm_head"))
    # KV cache side
    b.append(box(180, 340, 300, 56, "KV Cache", "per-layer K/V buffers · append", fs=12, fill="#1b2340", stroke="#5eead4"))
    b.append(arrow(330, 260, 330, 338, color="#5eead4", marker="arra", dashed=True))
    b.append('<text x="500" y="372" fill="#5eead4" font-size="11">incremental decode</text>')
    return svg_wrap("\n".join(b), 660, 420)

def svg_paged_kv():
    b = [SVG_DEFS]
    b.append('<text x="20" y="24" fill="#e9edf4" font-size="13" font-weight="700">Paged (vLLM-style)</text>')
    # block tables
    for i, (lab, n) in enumerate([("seq A · 40 tok", 3), ("seq B · 20 tok", 2), ("seq C · 61 tok", 4)]):
        y = 45 + i * 62
        b.append(f'<text x="20" y="{y+18}" fill="#99a4ba" font-size="12">{lab}</text>')
        b.append(f'<text x="150" y="{y+18}" fill="#6b7690" font-size="11">block table →</text>')
        for j in range(n):
            x = 250 + j * 46
            last = (j == n - 1)
            b.append(box(x, y, 42, 34, f"b{j}", "part" if last else "full", fs=11,
                         fill="#1a2b26" if last else "#131b28",
                         stroke="#3f7a68" if last else "#3b4d75"))
    b.append('<text x="20" y="250" fill="#99a4ba" font-size="11">waste = only the tail of the last block (≤ B−1 slots)</text>')
    # contiguous baseline
    b.append('<text x="20" y="292" fill="#e9edf4" font-size="13" font-weight="700">Contiguous baseline</text>')
    b.append(box(20, 308, 420, 34, "", "", fill="#131b28", stroke="#3b4d75"))
    b.append(f'<rect x="22" y="310" width="90" height="30" rx="6" fill="#1a2b26" stroke="#3f7a68"/>')
    b.append('<text x="67" y="330" text-anchor="middle" fill="#7ee2a8" font-size="11">128 used</text>')
    b.append('<text x="280" y="330" text-anchor="middle" fill="#f08787" font-size="11">1920 stranded</text>')
    b.append('<text x="20" y="366" fill="#99a4ba" font-size="11">reserves max_len per sequence: 81.5% waste on mixed workloads</text>')
    return svg_wrap("\n".join(b), 560, 390)

def svg_spec_decode():
    b = [SVG_DEFS]
    b.append(box(20, 90, 130, 60, "Draft model", "cheap · fast"))
    b.append(box(200, 90, 150, 60, "k draft tokens", "proposed", fs=12))
    b.append(arrow(150, 120, 198, 120))
    b.append(box(400, 90, 150, 60, "Target model", "one verify pass"))
    b.append(arrow(350, 120, 398, 120))
    b.append(box(400, 200, 150, 60, "Accept / reject", "per-token test", fs=12))
    b.append(arrow(475, 150, 475, 198))
    b.append(box(200, 200, 150, 60, "Output tokens", "accepted + bonus", fs=12))
    b.append(arrow(400, 230, 352, 230, color="#7ee2a8", marker="arrg"))
    b.append(arrow(275, 260, 85, 260, dashed=True, color="#5eead4", marker="arra"))
    b.append('<text x="180" y="252" fill="#5eead4" font-size="11">E = (1−αᵏ⁺¹)/(1−α) tokens per target pass</text>')
    b.append('<text x="180" y="282" fill="#99a4ba" font-size="11">speedup = E / (1 + k·c) — speculation is a bet on α</text>')
    return svg_wrap("\n".join(b), 580, 310)

def svg_disagg():
    b = [SVG_DEFS]
    b.append(box(20, 120, 120, 56, "Requests", "prompt + gen"))
    b.append(box(190, 40, 170, 60, "Prefill pool", "compute-optimized"))
    b.append(box(190, 200, 170, 60, "Decode pool", "bandwidth-optimized"))
    b.append(arrow(140, 148, 188, 80))
    b.append(arrow(140, 148, 188, 218))
    b.append(box(410, 110, 150, 60, "KV transfer", "bytes / bandwidth", fs=12, fill="#1b2340", stroke="#5eead4"))
    b.append(arrow(360, 70, 430, 118, color="#5eead4", marker="arra"))
    b.append(arrow(360, 230, 430, 152, color="#5eead4", marker="arra"))
    b.append(box(410, 200, 150, 60, "Tokens out", "TTFT · TBT", fs=12))
    b.append(arrow(560, 140, 560, 198))
    b.append('<text x="285" y="300" text-anchor="middle" fill="#99a4ba" font-size="11">wins when prefill stops blocking decodes; loses when transfer dominates</text>')
    return svg_wrap("\n".join(b), 600, 330)

def svg_quant():
    b = [SVG_DEFS]
    b.append(box(20, 100, 120, 56, "FP32 weights", "trained tiny LM"))
    schemes = [("INT8 ch.", "8.2 b/p"), ("FP8 E4M3", "8.0 b/p"), ("INT4 g32", "4.5 b/p"), ("INT4 g128", "4.2 b/p")]
    for i, (lab, sub) in enumerate(schemes):
        y = 20 + i * 62
        b.append(box(200, y, 130, 50, lab, sub, fs=12))
        b.append(arrow(140, 128, 198, y + 25))
    b.append(box(390, 100, 130, 56, "Eval ppl", "degradation", fs=12, fill="#1a2b26", stroke="#3f7a68"))
    for i in range(4):
        b.append(arrow(330, 45 + i * 62, 388, 118 + (i - 1.5) * 8))
    b.append('<text x="455" y="190" text-anchor="middle" fill="#7ee2a8" font-size="11">Pareto: bits vs quality</text>')
    b.append('<text x="285" y="300" text-anchor="middle" fill="#99a4ba" font-size="11">granularity tames outliers: per-channel/per-group ≫ per-tensor</text>')
    return svg_wrap("\n".join(b), 560, 330)

def svg_moe():
    b = [SVG_DEFS]
    b.append(box(20, 130, 110, 56, "Tokens", "batch"))
    b.append(box(180, 130, 130, 56, "Gating", "softmax top-k", fs=12))
    b.append(arrow(130, 158, 178, 158))
    # experts with load bars
    loads = [0.95, 0.8, 0.45, 0.3, 0.2, 0.12, 0.08, 0.05]
    for i, ld in enumerate(loads):
        x = 360 + (i % 4) * 78
        y = 60 + (i // 4) * 110
        b.append(box(x, y, 70, 70, f"E{i}", f"{ld:.0%} load", fs=11,
                     fill="#241d10" if ld > 0.9 else "#131b28",
                     stroke="#f0b35e" if ld > 0.9 else "#3b4d75"))
        b.append(arrow(310, 158, x + 10, y + 60, dashed=True))
    b.append(box(700, 130, 110, 56, "Combine", "weighted sum", fs=12))
    b.append(arrow(668, 120, 698, 140))
    b.append(arrow(668, 200, 698, 170))
    b.append('<text x="530" y="300" text-anchor="middle" fill="#f08787" font-size="11">capacity factor: tokens beyond capacity are DROPPED</text>')
    b.append('<text x="530" y="322" text-anchor="middle" fill="#99a4ba" font-size="11">expert-parallel all-to-all: total bytes independent of device count</text>')
    return svg_wrap("\n".join(b), 830, 350)

def svg_roofline():
    b = [SVG_DEFS]
    # axes
    b.append('<line x1="70" y1="20" x2="70" y2="280" stroke="#3b4d75" stroke-width="1.5"/>')
    b.append('<line x1="70" y1="280" x2="560" y2="280" stroke="#3b4d75" stroke-width="1.5"/>')
    b.append('<text x="30" y="150" fill="#99a4ba" font-size="11" transform="rotate(-90 30 150)" text-anchor="middle">FLOPS/s (log)</text>')
    b.append('<text x="315" y="305" fill="#99a4ba" font-size="11" text-anchor="middle">arithmetic intensity, FLOP/byte (log)</text>')
    # memory roof (diagonal) then compute roof (flat)
    b.append('<line x1="70" y1="280" x2="330" y2="80" stroke="#5eead4" stroke-width="2"/>')
    b.append('<line x1="330" y1="80" x2="560" y2="80" stroke="#5eead4" stroke-width="2"/>')
    b.append('<text x="180" y="200" fill="#5eead4" font-size="11">bandwidth roof</text>')
    b.append('<text x="445" y="65" fill="#5eead4" font-size="11">compute roof</text>')
    # op points: decode (low intensity) vs prefill (high)
    for x, y, lab, col in [(120, 265, "attn", "#f08787"), (150, 258, "qkv", "#f08787"),
                           (180, 250, "ffn", "#f08787"), (400, 95, "prefill gemm", "#7ee2a8")]:
        b.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{col}"/>')
        b.append(f'<text x="{x}" y="{y - 12}" text-anchor="middle" fill="{col}" font-size="10">{lab}</text>')
    b.append('<text x="315" y="335" text-anchor="middle" fill="#99a4ba" font-size="11">decode: memory-bound (left) · prefill: compute-bound (right)</text>')
    return svg_wrap("\n".join(b), 600, 360)

def svg_sched():
    b = [SVG_DEFS]
    b.append(box(20, 120, 120, 56, "Request queue", "Poisson arrivals"))
    policies = ["static-fcfs", "continuous", "chunked-prefill", "priority-preempt"]
    for i, pol in enumerate(policies):
        y = 20 + i * 60
        b.append(box(200, y, 170, 50, pol, "scheduler", fs=12,
                     fill="#1a2b26" if i == 1 else "#131b28",
                     stroke="#3f7a68" if i == 1 else "#3b4d75"))
        b.append(arrow(140, 148, 198, y + 25))
    b.append(box(430, 100, 140, 90, "GPU iterations", "t = T0 + K·batch tok", fs=12))
    for i in range(4):
        b.append(arrow(370, 45 + i * 60, 428, 120 + (i - 1.5) * 12))
    b.append(box(430, 230, 140, 56, "Completions", "TTFT · TPOT", fs=12))
    b.append(arrow(500, 190, 500, 228))
    b.append('<text x="300" y="330" text-anchor="middle" fill="#99a4ba" font-size="11">iteration-level admission is the dominant win (42x p99 TTFT)</text>')
    return svg_wrap("\n".join(b), 600, 360)

def svg_remit():
    b = [SVG_DEFS]
    b.append(box(230, 14, 140, 52, "Declared remit R", "purpose · authority · limits", fs=12,
                 fill="#1a2b26", stroke="#3f7a68"))
    types = ["functional", "authority", "goal", "planning",
             "tool", "information", "collaboration", "temporal"]
    for i, tp in enumerate(types):
        row, col = divmod(i, 4)
        x, y = 20 + col * 145, 110 + row * 64
        b.append(box(x, y, 130, 50, tp, "D(t) grows", fs=11))
        b.append(arrow(300, 66, x + 65, y, color="#f08787", marker="arr"))
    b.append(box(180, 252, 240, 56, "Policy engine", "allow · deny · escalate", fs=13,
                 fill="#1a2436", stroke="#6ea8fe"))
    b.append(arrow(300, 238, 300, 250, color="#6ea8fe", marker="arra"))
    b.append('<text x="300" y="336" text-anchor="middle" fill="#99a4ba" font-size="11">D(t) = Distance(A(t), R): validated before execution, divergence stays bounded</text>')
    return svg_wrap("\n".join(b), 600, 360)

def svg_mesh():
    b = [SVG_DEFS]
    b.append(box(230, 14, 140, 52, "Semantic router", "task type → best model", fs=12,
                 fill="#1a2b26", stroke="#3f7a68"))
    models = [("frontier-1", "generalist"), ("reasoner-r1", "reasoning"),
              ("coder-cl", "coding"), ("vision-qw", "vision"), ("general-8b", "cheap chat")]
    for i, (m, role) in enumerate(models):
        x = 14 + i * 116
        b.append(box(x, 110, 106, 54, m, role, fs=10))
        b.append(arrow(300, 66, x + 53, 108, color="#6ea8fe", marker="arra"))
    b.append(box(180, 208, 240, 56, "Orchestrator", "parallel fan-out · aggregate · failover", fs=12,
                 fill="#1a2436", stroke="#6ea8fe"))
    for i in range(5):
        x = 14 + i * 116 + 53
        b.append(arrow(x, 164, 300, 206, color="#99a4ba", marker="arra"))
    b.append('<text x="300" y="300" text-anchor="middle" fill="#99a4ba" font-size="11">UCB1 router learns per-task winners; quality 0.943 at 0.38x the cost of one frontier model</text>')
    b.append('<text x="300" y="322" text-anchor="middle" fill="#99a4ba" font-size="11">model outage → automatic failover, success stays 1.000</text>')
    return svg_wrap("\n".join(b), 600, 360)

# ---------------------------------------------------------------------------
# README parsing (Problem / Current gap / Solution / Insights)
# ---------------------------------------------------------------------------

def md_inline(s):
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s

def md_block(text):
    """Convert a README section (paragraphs + -/* lists + numbered lists) to HTML."""
    html = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(md_inline(re.sub(r"^[-*]\s+", "", lines[i].strip())))
                i += 1
                while i < len(lines) and lines[i].startswith(" ") and lines[i].strip():
                    items[-1] += " " + md_inline(lines[i].strip())
                    i += 1
            html.append("<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>")
        elif re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s+", lines[i].strip()):
                items.append(md_inline(re.sub(r"^\d+\.\s+", "", lines[i].strip())))
                i += 1
                while i < len(lines) and lines[i].startswith(" ") and lines[i].strip():
                    items[-1] += " " + md_inline(lines[i].strip())
                    i += 1
            html.append("<ol>" + "".join(f"<li>{x}</li>" for x in items) + "</ol>")
        else:
            para = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^([-*]|\d+\.)\s+", lines[i].strip()):
                para.append(lines[i].strip())
                i += 1
            html.append(f"<p>{md_inline(' '.join(para))}</p>")
    return "\n".join(html)

def read_sections(repo_dir):
    text = open(os.path.join(ROOT, repo_dir, "README.md")).read()
    secs = {}
    for name in ["Problem", "Current gap", "Solution", "Insights"]:
        m = re.search(r"## " + name + r"\n(.*?)(?=\n## |\Z)", text, re.S)
        secs[name] = md_block(m.group(1).strip()) if m else ""
    return secs

# ---------------------------------------------------------------------------
# Project content (results tables use verified measured numbers)
# ---------------------------------------------------------------------------

PROJECTS = [
dict(
 slug="nano-transformer", num="01", name="Nano Transformer",
 tagline="A LLaMA-style decoder inference engine from scratch in numpy — verified bit-exact",
 category="foundations", repo="01-nano-transformer", arch=svg_transformer,
 metric="<b>0.0</b> max diff KV-cache vs full forward · <b>137x</b> decode speedup @512",
 impl_tree="""src/llm_inference_engine/transformer/
  model.py          # config, RMSNorm, RoPE, GQA, SwiGLU, KVCache, generate
01-nano-transformer/
  experiments/run_experiments.py   # 4 verifications + latency benchmarks
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>Check</th><th>Max abs diff</th><th>Verdict</th></tr>
<tr><td>RMSNorm vs naive loop</td><td>4.77e-07</td><td>match</td></tr>
<tr><td>RoPE vs complex-exponential reference</td><td>2.38e-07</td><td>match</td></tr>
<tr><td>GQA vs tied full-head attention</td><td>0.0</td><td><b class="hl">bit-exact</b></td></tr>
<tr><td>KV-cache decode vs full forward</td><td>0.0</td><td><b class="hl">bit-exact</b></td></tr></table>
<table class="res"><tr><th>Seq len</th><th>Decode cached</th><th>Decode recompute</th><th>Speedup</th></tr>
<tr><td>32</td><td>~1 ms/tok</td><td>~21 ms/tok</td><td>11.5x</td></tr>
<tr><td>512</td><td>4.72 ms/tok</td><td>645.68 ms/tok</td><td><b class="hl">136.8x</b></td></tr></table>
<p>KV-cache memory (fp16, analytic): 16 KiB at 32 tokens → 256 KiB at 512. Prefill: 13.5 ms at 32 → 557.3 ms at 512.</p>""",
),
dict(
 slug="paged-kv", num="02", name="Paged KV-Cache",
 tagline="vLLM-style paged block manager: quantifying fragmentation waste vs contiguous allocation",
 category="serving", repo="02-paged-kv", arch=svg_paged_kv,
 metric="<b>1.9%</b> waste vs <b>81.5%</b> contiguous · <b>817</b> vs 488 concurrent",
 impl_tree="""src/llm_inference_engine/paged_kv/
  blocks.py         # BlockManager (paged) + ContiguousAllocator (baseline)
  workload.py       # mixed sequence-length request stream
02-paged-kv/
  experiments/run_experiments.py   # 2000-request sim + block-size sweep
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>Allocator</th><th>Waste</th><th>Max concurrent</th><th>Rejected / 2000</th></tr>
<tr><td><b>Paged B=16</b></td><td><b class="hl">1.90%</b></td><td><b class="hl">817</b></td><td><b>0</b></td></tr>
<tr><td>Contiguous (max_len reserve)</td><td>81.51%</td><td>488</td><td>603</td></tr></table>
<table class="res"><tr><th>Block size</th><th>8</th><th>16</th><th>32</th><th>64</th></tr>
<tr><td>Paged waste</td><td>0.87%</td><td>1.90%</td><td>3.88%</td><td>7.50%</td></tr></table>
<p>1M-slot budget, 2000 requests, mean length 373 tokens (seed 11, ran twice identical). Paging converts a capacity problem into a rounding problem: waste is bounded by B−1 slots per sequence.</p>""",
),
dict(
 slug="spec-decode", num="03", name="Speculative Decoding",
 tagline="Draft + target speculative sampling: an analytical model validated by Monte Carlo",
 category="serving", repo="03-spec-decode", arch=svg_spec_decode,
 metric="<b>5.28x</b> best speedup · formula vs MC dev <b>0.018</b> tokens",
 impl_tree="""src/llm_inference_engine/spec_decode/
  model.py          # E(k,alpha), speedup, optimal-k selector
  sim.py            # vectorized Monte Carlo of the accept/reject loop
03-spec-decode/
  experiments/run_experiments.py   # formula validation + k/alpha/c sweep
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>Scenario (c=0.1)</th><th>Optimal k</th><th>Speedup</th><th>tok/s vs 25.0 vanilla</th></tr>
<tr><td>alpha=0.95</td><td>8</td><td><b class="hl">5.28x</b></td><td>—</td></tr>
<tr><td>alpha=0.70</td><td>4</td><td>1.98x</td><td>49.5</td></tr>
<tr><td>alpha=0.50</td><td>2</td><td>1.46x</td><td>36.5</td></tr>
<tr><td>alpha=0.30</td><td>1</td><td>1.18x</td><td>29.5</td></tr>
<tr><td>alpha=0.30, k=8 (too aggressive)</td><td>—</td><td><b>0.79x (slowdown)</b></td><td>19.8</td></tr></table>
<p>Analytical model E = (1−α<sup>k+1</sup>)/(1−α), speedup = E/(1+k·c); validated against 120 Monte Carlo cells × 100k trials (max deviation 0.018 tokens). The selector never recommends a slowdown.</p>""",
),
dict(
 slug="disagg-serve", num="04", name="Disaggregated Serving",
 tagline="Prefill/decode disaggregation simulator: TTFT vs transfer-cost tradeoffs, quantified",
 category="serving", repo="04-disagg-serve", arch=svg_disagg,
 metric="<b>2.4x</b> lower TTFT on long prompts · crossover at <b>~81</b> tokens",
 impl_tree="""src/llm_inference_engine/disagg_serve/
  sim.py            # discrete-event sim: colocated vs disaggregated
04-disagg-serve/
  experiments/run_experiments.py   # workload grid + crossover brackets
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>Workload (high load)</th><th>Colocated TTFT</th><th>Disaggregated TTFT</th></tr>
<tr><td>Long prompts, mean</td><td>40.4 ms</td><td><b class="hl">17.1 ms</b></td></tr>
<tr><td>Long prompts, p99</td><td>382.5 ms</td><td><b class="hl">166.2 ms</b></td></tr>
<tr><td>Short prompts, mean</td><td>7.1 ms</td><td><b>4.9 ms</b></td></tr></table>
<p>Disaggregation starts winning at <b>~81-token prompts</b>; below that the KV-transfer tax dominates. The win is erased below <b>~74 GB/s</b> interconnect (at 5 GB/s, disaggregated TTFT is ~10x worse). Two seeded runs byte-identical.</p>""",
),
dict(
 slug="quant-lab", num="05", name="Quantization Lab",
 tagline="INT8/FP8/INT4 workbench with real perplexity degradation on a trained tiny LM",
 category="efficiency", repo="05-quant-lab", arch=svg_quant,
 metric="INT8 <b>lossless</b> (2.1131 vs 2.1130) · INT4-g32 <b>+0.036</b> @ 4.5 bits",
 impl_tree="""src/llm_inference_engine/quant_lab/
  schemes.py        # INT8 per-channel/per-tensor, FP8 E4M3, INT4 per-group
  tiny_lm.py        # from-scratch char LM (numpy, Adam)
  eval.py           # perplexity under quantized weights
05-quant-lab/
  experiments/run_experiments.py   # train once, eval twice, sensitivity
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>Scheme</th><th>Eval perplexity</th><th>Bits/param</th></tr>
<tr><td>Baseline (fp32)</td><td>2.1130</td><td>32</td></tr>
<tr><td><b>INT8 per-channel</b></td><td><b class="hl">2.1131</b></td><td>8.22</td></tr>
<tr><td>FP8 E4M3</td><td>2.1212</td><td>8.00</td></tr>
<tr><td>INT4 group-32</td><td>2.1490</td><td>4.50</td></tr>
<tr><td>INT4 group-128</td><td>2.1867</td><td>4.22</td></tr></table>
<p>74k-param char transformer trained from scratch (600 Adam steps, seed 41); eval run twice bit-identical. On outlier-structured synthetic weights, per-tensor INT8 has ~8x worse reconstruction error than per-channel — granularity exists to tame outliers.</p>""",
),
dict(
 slug="moe-router", num="06", name="MoE Router",
 tagline="Expert routing simulator: capacity factor, token dropping, expert parallelism",
 category="efficiency", repo="06-moe-router", arch=svg_moe,
 metric="Balanced gating drops <b>9.3%</b> vs <b>61%</b> skewed @CF=1.0",
 impl_tree="""src/llm_inference_engine/moe_router/
  gating.py         # top-k, noisy, and balanced gating
  router.py         # capacity-limited dispatch + drop accounting
  parallel.py       # expert-parallel sharding + all-to-all cost
06-moe-router/
  experiments/run_experiments.py   # CF sweep + EP scaling
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>top-1, CF=1.0</th><th>Drop rate</th><th>Imbalance (max/mean)</th></tr>
<tr><td>Skewed popularity</td><td><b>61.0%</b></td><td>2.56</td></tr>
<tr><td><b>Balanced (aux-loss)</b></td><td><b class="hl">9.3%</b></td><td><b>1.10</b></td></tr></table>
<p>4096 tokens, 8 experts, seed 51, 4 runs identical. Skewed gating at CF=2.0 still drops 38.1% — balancing beats raw capacity. Expert parallelism (16 experts, 8 devices): 16.0 MiB/device all-to-all per step; total comm independent of device count.</p>""",
),
dict(
 slug="roofline", num="07", name="Roofline Analysis",
 tagline="Per-op roofline model for inference: where the wall is, and what fusion moves",
 category="foundations", repo="07-roofline", arch=svg_roofline,
 metric="Decode <b>7/7 memory-bound</b> @0.65% peak · fused dequant+GEMM <b>1.83x</b>",
 impl_tree="""src/llm_inference_engine/roofline/
  hardware.py       # A100 / H100 / CPU published specs
  ops.py            # per-op FLOP/byte accounting for a decoder layer
  model.py          # classification + fusion transforms
07-roofline/
  experiments/run_experiments.py   # decode vs prefill + fusion estimates
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>Regime (A100, 7B-ish, S=2048)</th><th>Memory-bound</th><th>Step time</th><th>Dominant op</th></tr>
<tr><td><b>Decode B=1</b></td><td><b class="hl">7/7</b> (~0.65% of peak)</td><td>181.3 µs</td><td>ffn_gate_up 36.3%</td></tr>
<tr><td>Prefill B=8</td><td>3/7 (GEMMs compute-bound @100%)</td><td>22.10 ms</td><td>ffn_gate_up 31.9%</td></tr></table>
<p>FlashAttention-style fusion: 1.001x decode (nothing to kill at B=1), <b>1.139x prefill</b> (attention intensity 112.7 → 1034 FLOP/byte). Fused INT8 dequant+GEMM: <b>1.831x decode step</b>. Deterministic; two runs identical.</p>""",
),
dict(
 slug="sched-lab", num="08", name="Scheduling Lab",
 tagline="Continuous batching vs chunked prefill vs preemption: a scheduling shootout",
 category="serving", repo="08-sched-lab", arch=svg_sched,
 metric="<b>56x</b> lower mean TTFT continuous vs static · <b>82.8%</b> lower worst-case iter",
 impl_tree="""src/llm_inference_engine/sched_lab/
  sim.py            # iteration-stepped discrete-event simulator, 4 policies
08-sched-lab/
  experiments/run_experiments.py   # load sweep + burst + preemption cost
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>High load (6 req/s)</th><th>Mean TTFT</th><th>p99 TTFT</th><th>Throughput</th></tr>
<tr><td>static-fcfs</td><td>844.6 ms</td><td>2437.1 ms</td><td>1861 tok/s</td></tr>
<tr><td><b>continuous</b></td><td><b class="hl">15.0 ms</b></td><td><b class="hl">57.5 ms</b></td><td>1861 tok/s</td></tr></table>
<p>1500 requests/cell, seed 61, 3 runs byte-identical. Scheduling moves latency, not capacity: throughput is identical. Chunked prefill bounds worst-case iteration latency to 25.7 ms vs 149.5 ms (−82.8%). Priority preemption: 14 preempts cost 4.88% recompute for −87.9% short-request p99.</p>""",
),
dict(
 slug="remit-drift", num="09", name="Remit Drift",
 tagline="Divergence of remit in multi-agent systems: measured per type, governed by a policy engine",
 category="governance", repo="09-remit-drift", arch=svg_remit,
 metric="All <b>8 drift types</b> grow ungoverned · <b>0.000</b> max D(t) governed",
 impl_tree="""src/llm_inference_engine/remit_drift/
  remit.py          # Remit, Grant: the mandate as a runtime object
  policy.py         # PolicyEngine: allow/deny/escalate, 9 controls
  metrics.py        # D(t): KL, violation rates, proxy/true gap
  sim.py            # drift emerges from pressure + creep
  scenarios.py      # 8 specialists, finance agent, support demo
09-remit-drift/
  experiments/run_experiments.py   # 5 experiments, seeded (seed 11)
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>Divergence type</th><th>D(300) ungoverned</th><th>D(300) governed</th></tr>
<tr><td>functional (KL)</td><td>5.550</td><td><b class="hl">0.000</b></td></tr>
<tr><td>authority</td><td>1.000</td><td><b class="hl">0.000</b></td></tr>
<tr><td>goal (proxy/true gap)</td><td>0.800</td><td><b class="hl">0.000</b></td></tr>
<tr><td>planning</td><td>1.000</td><td><b class="hl">0.000</b></td></tr>
<tr><td>tool</td><td>1.000</td><td><b class="hl">0.000</b></td></tr>
<tr><td>information</td><td>1.000</td><td><b class="hl">0.000</b></td></tr>
<tr><td>collaboration</td><td>0.680</td><td><b class="hl">0.000</b></td></tr>
<tr><td>temporal</td><td>1.000</td><td><b class="hl">0.000</b></td></tr></table>
<table class="res"><tr><th>Scenario</th><th>Ungoverned</th><th>Governed</th></tr>
<tr><td>Finance agent: first out-of-remit use</td><td>steps 16 / 26 / 38</td><td><b class="hl">0 executed</b> (175 denied)</td></tr>
<tr><td>Support tickets: close-now rate</td><td>0.71, satisfaction 0.38</td><td><b class="hl">0.00</b>, satisfaction <b class="hl">0.95</b></td></tr></table>
<p>Seed 11, T=300 steps, two runs byte-identical. Ablation: authorization checks catch 4 types, human approval 3, tool allow-lists / planner constraints / policy engine 1-2 each; auditing detects 7 of 8 (detect-only). A written remit alone does <b>not</b> contain functional drift — repeated denied requests are silently absorbed (normalization of deviance); central enforcement plus versioning is required.</p>""",
),
dict(
 slug="llm-mesh", num="10", name="LLM Mesh",
 tagline="Many specialized models, one routing layer: route, orchestrate, aggregate",
 category="serving", repo="10-llm-mesh", arch=svg_mesh,
 metric="<b>0.943</b> quality at <b>0.38x</b> the cost of a single frontier model",
 impl_tree="""src/llm_inference_engine/llm_mesh/
  models.py         # registry of 5 specialists: quality/latency/price profiles
  router.py         # TF-IDF semantic router + 4 routing policies
  orchestrator.py   # decomposition, parallel dispatch, failover
  bandit.py         # UCB1 self-improving router
  observability.py  # per-request traces: latency, tokens, cost
  workflows.py      # fraud-claim + chart+report pipelines
10-llm-mesh/
  experiments/run_experiments.py   # 6 experiments, seeded (seed 7)
  results/            # results.md + charts.png (generated)""",
 results_table="""
<table class="res"><tr><th>400 mixed requests</th><th>Mean quality</th><th>Total cost</th><th>p95 latency</th></tr>
<tr><td>frontier-1 only</td><td>0.9150</td><td>$5.00</td><td>4956 ms</td></tr>
<tr><td><b>mesh (quality-first)</b></td><td><b class="hl">0.9430</b></td><td><b class="hl">$1.92</b></td><td>6395 ms</td></tr></table>
<table class="res"><tr><th>Routing policy</th><th>Quality</th><th>Cost</th></tr>
<tr><td>quality-first</td><td>0.944</td><td>$1.92</td></tr>
<tr><td>balanced</td><td>0.847</td><td>$0.43</td></tr>
<tr><td>latency-first</td><td>0.783</td><td>$0.33</td></tr>
<tr><td>cost-first</td><td>0.782</td><td>$0.33</td></tr></table>
<p>Seed 7, 3 runs byte-identical. Semantic router: 1.0000 accuracy on 24 held-out queries. Parallel orchestration: 1.64x speedup (p95 6947 ms → 4486 ms). UCB1 regret 0.1566 → 0.0296 over 3000 requests. Failover: 1.000 vs 0.750 success when vision-qw goes down. The quality win is honest about its price: mesh p95 latency is higher because quality-first routes reasoning to the slowest specialist — the routing policy chooses which axis you optimize.</p>""",
),
]

CATEGORIES = [
dict(slug="foundations", name="From First Principles",
     color="#5eead4",
     desc="Understand inference at the metal level: build the transformer yourself, then learn where the hardware wall actually is."),
dict(slug="serving", name="Serving at Scale",
     color="#6ea8fe",
     desc="The serving stack, layer by layer: memory management, latency hiding, architecture, and scheduling — each quantified."),
dict(slug="efficiency", name="Efficiency Frontiers",
     color="#f0b35e",
     desc="Do more with less: squeeze quality out of fewer bits, and capacity out of sparse experts."),
dict(slug="governance", name="Agent Governance",
     color="#f08787",
     desc="The systems around the model: when agents drift from their remit, measure it per type and contain it with a policy engine."),
]

# ---------------------------------------------------------------------------
# Templates + generation
# ---------------------------------------------------------------------------

def nav(prefix=""):
    cats = "".join(
        f'<a href="{prefix}index.html#{c["slug"]}">{c["name"]}</a>'
        for c in CATEGORIES)
    return f"""
<nav class="top"><div class="wrap">
<a class="brand" href="{prefix}index.html">LLM Inference <span>Engine</span></a>
<div class="links">{cats}<a href="https://github.com/ranadhir19/llm-inference-engine/blob/main/docs/user-guide.md">User Guide ↗</a><a href="https://github.com/ranadhir19/llm-inference-engine">GitHub ↗</a></div>
</div></nav>"""

def page_head(title):
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — LLM Inference Engine</title>
<style>{CSS}</style></head><body>"""

PAGE_FOOT = """
<footer><div class="wrap">
<div>LLM Inference Engine — ten advanced inference systems from first principles,<br>implemented in numpy, benchmarked, and reproducible.</div>
<div>Built by Ranadhir Ghosh · MIT License</div>
</div></footer></body></html>"""

def index_page():
    cats_html = ""
    for c in CATEGORIES:
        projs = [p for p in PROJECTS if p["category"] == c["slug"]]
        cards = ""
        for p in projs:
            secs = read_sections(p["repo"])
            teaser = re.sub(r"<[^>]+>", "", secs["Problem"])[:170].replace("\n", " ")
            cards += f"""
<div class="card"><div class="tag">{p["num"]} · {p["name"]}</div>
<h4><a href="{p["slug"]}/index.html">{p["tagline"]}</a></h4>
<p>{teaser}…</p>
<div class="metric">{p["metric"]}</div></div>"""
        cats_html += f"""
<div class="cat" id="{c["slug"]}">
<h3><span class="dot" style="background:{c["color"]}"></span>{c["name"]}</h3>
<p class="desc">{c["desc"]}</p>
<div class="cards">{cards}</div></div>"""
    return (page_head("LLM Inference Engine") + nav() + f"""
<div class="hero"><div class="wrap">
<div class="kicker">Portfolio · LLM Inference Systems</div>
<h1>Nine systems for how LLMs actually run.</h1>
<p class="sub">From a transformer built from scratch in numpy to the scheduling
policies that decide what users feel — each project implements an advanced
inference technique, benchmarks it with seeded experiments, and reports
honest numbers. No GPUs required. No hand-waving.</p>
<a class="btn" href="#foundations">Explore the systems</a>
<a class="btn ghost" href="nano-transformer/index.html">Start from first principles →</a>
<div class="install"><code>pip install llm-inference-lab</code><span>The full library — ten systems, one import.</span></div>
<div class="stats">
<div><b>10</b><span>implemented systems</span></div>
<div><b>4</b><span>categories</span></div>
<div><b>30+</b><span>experiments, all seeded</span></div>
<div><b>100%</b><span>reproducible — one command each</span></div>
</div></div></div>
<section><div class="wrap">
<h2 class="sec">Browse by category</h2>
<p class="lead">From the math of a single decode step to the scheduling of a
thousand concurrent requests. Pick a layer, then drill into any system.</p>
{cats_html}
</div></section>
<section><div class="wrap">
<h2 class="sec">How to read a project</h2>
<p class="lead">Every project page follows the same spine:</p>
<div class="cards">
<div class="card"><h4>1 · Problem &amp; gap</h4><p>What breaks or costs too much in real inference stacks, and why the standard treatments don't settle it.</p></div>
<div class="card"><h4>2 · Solution &amp; architecture</h4><p>The implemented technique, with a blueprint-style architecture diagram.</p></div>
<div class="card"><h4>3 · Results &amp; insights</h4><p>Measured numbers from seeded experiments, plus what the numbers actually mean.</p></div>
<div class="card"><h4>4 · Implementation</h4><p>The code layout — everything is implemented, tested, and reproducible.</p></div>
</div></div></section>
""" + PAGE_FOOT)

def project_page(p, prev_p, next_p, cat):
    secs = read_sections(p["repo"])
    return (page_head(p["name"]) + nav("../") + f"""
<div class="wrap"><div class="proj-hero">
<div class="crumb"><a href="../index.html">Lab</a> / <a href="../index.html#{cat["slug"]}">{cat["name"]}</a> / {p["num"]}</div>
<div class="kicker" style="color:{cat["color"]}">{cat["name"]}</div>
<h1>{p["name"]}</h1>
<p class="tagline">{p["tagline"]}</p>
</div>
<div class="block"><h2><span class="n">1</span>The problem</h2>{secs["Problem"]}</div>
<div class="block"><h2><span class="n">2</span>The current gap</h2>{secs["Current gap"]}</div>
<div class="block"><h2><span class="n">3</span>How the solution addresses it</h2>{secs["Solution"]}</div>
<div class="block"><h2><span class="n">4</span>Architecture</h2>
<div class="diagram">{p["arch"]()}</div>
<div class="caption">Blueprint of the {p["name"]} architecture. See the repo for the full implementation.</div></div>
<div class="block"><h2><span class="n">5</span>Results</h2>
{p["results_table"]}
<img class="chart" src="../assets/charts/{p["slug"]}.png" alt="{p["name"]} experiment charts">
<div class="caption">Generated by <code>experiments/run_experiments.py</code> — re-running reproduces these exactly.</div></div>
<div class="block"><h2><span class="n">6</span>Insights</h2>{secs["Insights"]}</div>
<div class="block"><h2><span class="n">7</span>Implementation</h2>
<code class="tree">{p["impl_tree"]}</code>
<p style="margin-top:14px">Full source, tests, and experiment code: <a href="https://github.com/ranadhir19/llm-inference-engine/tree/main/{p["repo"]}/">{p["repo"]}/</a> ·
Run: <code>cd {p["repo"]} && python3 experiments/run_experiments.py</code></p></div>
<div class="pagenav">
<a href="../{prev_p["slug"]}/index.html"><small>← Previous</small>{prev_p["num"]} · {prev_p["name"]}</a>
<a href="../{next_p["slug"]}/index.html" style="text-align:right"><small>Next →</small>{next_p["num"]} · {next_p["name"]}</a>
</div></div>
""" + PAGE_FOOT)

def main():
    charts_dir = os.path.join(HERE, "assets", "charts")
    os.makedirs(charts_dir, exist_ok=True)
    # copy charts from each project's results/
    for p in PROJECTS:
        src = os.path.join(ROOT, p["repo"], "results", "charts.png")
        dst = os.path.join(charts_dir, p["slug"] + ".png")
        if os.path.isfile(src):
            shutil.copyfile(src, dst)
        else:
            print(f"WARNING: missing chart {src}")
    with open(os.path.join(HERE, "index.html"), "w") as f:
        f.write(index_page())
    for i, p in enumerate(PROJECTS):
        d = os.path.join(HERE, p["slug"])
        os.makedirs(d, exist_ok=True)
        cat = next(c for c in CATEGORIES if c["slug"] == p["category"])
        prev_p = PROJECTS[(i - 1) % len(PROJECTS)]
        next_p = PROJECTS[(i + 1) % len(PROJECTS)]
        with open(os.path.join(d, "index.html"), "w") as f:
            f.write(project_page(p, prev_p, next_p, cat))
    print(f"built index + {len(PROJECTS)} project pages into {HERE}")

if __name__ == "__main__":
    main()
