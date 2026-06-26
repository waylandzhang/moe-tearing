#!/usr/bin/env python3
"""Generate paper-draft SVG figures from result JSON files.

This intentionally uses only the Python standard library so the plotting path is
reproducible on a fresh GPU box or laptop without matplotlib.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "gray": "#555555",
}


def load_json(name: str):
    return json.loads((ROOT / name).read_text())


def metric(d: dict, key: str, default=0.0):
    val = d.get(key, default)
    return default if val is None else val


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def poly(points) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


class Panel:
    def __init__(self, x, y, w, h, xmin, xmax, ymin, ymax, title):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.title = title

    def sx(self, v):
        if self.xmax <= self.xmin:
            return self.x + self.w / 2
        return self.x + (v - self.xmin) / (self.xmax - self.xmin) * self.w

    def sy(self, v):
        if self.ymax <= self.ymin:
            return self.y + self.h / 2
        return self.y + self.h - (v - self.ymin) / (self.ymax - self.ymin) * self.h

    def axes(self, xticks=5, yticks=4, xlabel=None):
        out = [
            f'<rect x="{self.x}" y="{self.y}" width="{self.w}" height="{self.h}" fill="#fff" stroke="#d0d0d0"/>',
            f'<text x="{self.x}" y="{self.y - 16}" font-size="20" font-weight="700" fill="#111">{esc(self.title)}</text>',
        ]
        for i in range(xticks + 1):
            xx = self.x + self.w * i / xticks
            val = self.xmin + (self.xmax - self.xmin) * i / xticks
            out.append(f'<line x1="{xx:.1f}" y1="{self.y}" x2="{xx:.1f}" y2="{self.y + self.h}" stroke="#eee"/>')
            out.append(f'<text x="{xx:.1f}" y="{self.y + self.h + 24}" text-anchor="middle" font-size="13" fill="#444">{val:.0f}</text>')
        for i in range(yticks + 1):
            yy = self.y + self.h * i / yticks
            val = self.ymax - (self.ymax - self.ymin) * i / yticks
            out.append(f'<line x1="{self.x}" y1="{yy:.1f}" x2="{self.x + self.w}" y2="{yy:.1f}" stroke="#eee"/>')
            out.append(f'<text x="{self.x - 10}" y="{yy + 5:.1f}" text-anchor="end" font-size="13" fill="#444">{val:.2f}</text>')
        if xlabel:
            out.append(f'<text x="{self.x + self.w/2}" y="{self.y + self.h + 50}" text-anchor="middle" font-size="15" fill="#333">{esc(xlabel)}</text>')
        return out

    def line(self, xs, ys, color, width=2.2, opacity=0.95):
        pts = [(self.sx(x), self.sy(y)) for x, y in zip(xs, ys)]
        return f'<polyline points="{poly(pts)}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"/>'

    def points(self, xs, ys, color, r=3.5, opacity=0.9):
        return [
            f'<circle cx="{self.sx(x):.1f}" cy="{self.sy(y):.1f}" r="{r}" fill="{color}" opacity="{opacity}"/>'
            for x, y in zip(xs, ys)
        ]

    def bars(self, xs, ys, color, width=36, opacity=0.85):
        out = []
        zero = self.sy(0)
        for x, y in zip(xs, ys):
            xx = self.sx(x) - width / 2
            yy = min(self.sy(y), zero)
            hh = abs(zero - self.sy(y))
            out.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{width}" height="{hh:.1f}" fill="{color}" opacity="{opacity}"/>')
        return out


def write_svg(name: str, width: int, height: int, elements: list[str]):
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        *elements,
        "</svg>",
    ]
    path = FIG / name
    path.write_text("\n".join(svg))
    print(f"wrote {path.relative_to(ROOT)}")


def tear_records():
    data = load_json("run_e64_hard.json")
    rows = []
    for r in data["log"]:
        if "tear" in r:
            t = r["tear"]
            rows.append({
                "step": r["step"],
                "loss": r["loss"],
                "tear_med": t["tear_med"],
                "hardG": t["hardG"],
                "hardJump": t["hardJump"],
                "softJump": t["softJump"],
            })
    return rows, data["summary"]


def fig_e64_training():
    rows, summary = tear_records()
    xs = [r["step"] for r in rows]
    tear = [r["tear_med"] for r in rows]
    hardg = [r["hardG"] for r in rows]
    jump = [r["hardJump"] for r in rows]

    w, h = 1180, 780
    p1 = Panel(90, 70, 480, 250, min(xs), max(xs), min(tear) - 0.01, max(tear) + 0.01, "M2 expert cliff stays flat")
    p2 = Panel(670, 70, 420, 250, min(xs), max(xs), 14.5, 17.5, "Continuity signature stays torn")
    p3 = Panel(90, 430, 1000, 250, min(xs), max(xs), 0.0, max(jump) * 1.15, "Whole-block jump collapses early, then plateaus")
    els = [f'<text x="{w/2}" y="30" text-anchor="middle" font-size="24" font-weight="700">E=64/k=8 natural training: topology persists, operational jump shrinks</text>']
    els += p1.axes(xlabel="step") + [p1.line(xs, tear, COLORS["blue"])] + p1.points(xs, tear, COLORS["blue"])
    els += p2.axes(xlabel="step") + [p2.line(xs, hardg, COLORS["purple"])] + p2.points(xs, hardg, COLORS["purple"])
    els += p3.axes(xlabel="step") + [p3.line(xs, jump, COLORS["orange"])] + p3.points(xs, jump, COLORS["orange"])
    els.append(f'<text x="90" y="735" font-size="15" fill="#333">summary: max spike={summary["max_spike"]:.3f}, spikes &gt;0.3={summary["n_spikes_gt_0.3"]}, diverged={summary.get("diverged", False)}</text>')
    write_svg("fig_e64_training_tear.svg", w, h, els)


def fig_e64_loss_spikes():
    data = load_json("run_e64_hard.json")
    log = [r for r in data["log"] if r.get("loss") is not None]
    xs = [r["step"] for r in log]
    loss = [r["loss"] for r in log]
    spikes = [r.get("spike", 0.0) for r in log]
    w, h = 1180, 680
    p1 = Panel(90, 70, 1000, 230, min(xs), max(xs), min(loss), max(loss), "Training loss")
    p2 = Panel(90, 390, 1000, 190, min(xs), max(xs), 0.0, max(spikes) * 1.1, "Positive loss jumps (spikes)")
    els = [f'<text x="{w/2}" y="30" text-anchor="middle" font-size="24" font-weight="700">E=64/k=8 training is spiky but convergent</text>']
    els += p1.axes(xlabel="step") + [p1.line(xs, loss, COLORS["blue"], width=1.5)]
    els += p2.axes(xlabel="step") + [p2.line(xs, spikes, COLORS["red"], width=1.4)]
    spike_x = [x for x, sp in zip(xs, spikes) if sp > 0.3]
    spike_y = [sp for sp in spikes if sp > 0.3]
    els += p2.points(spike_x, spike_y, COLORS["red"], r=2.4, opacity=0.7)
    write_svg("fig_e64_loss_spikes.svg", w, h, els)


def fig_tear_level_seed_bars():
    files = [
        ("seed0", 0.0, "run_e64_tear0.0.json"),
        ("seed0", 0.5, "run_e64_tear0.5.json"),
        ("seed0", 1.0, "run_e64_tear1.0.json"),
        ("seed1", 0.0, "run_e64_tear0.0_seed1.json"),
        ("seed1", 0.5, "run_e64_tear0.5_seed1.json"),
        ("seed1", 1.0, "run_e64_tear1.0_seed1.json"),
    ]
    rows = []
    for seed, tl, path in files:
        s = load_json(path)["summary"]
        rows.append({
            "seed": seed,
            "tear": tl,
            "max_spike": s["max_spike"],
            "n_spikes": s["n_spikes_gt_0.3"],
            "diverged": s.get("diverged", False),
        })
    w, h = 1100, 620
    xs = list(range(len(rows)))
    labels = [f'{r["seed"]}\ntl={r["tear"]}' for r in rows]
    p1 = Panel(90, 70, 440, 360, -0.6, len(rows) - 0.4, 0, max(r["n_spikes"] for r in rows) * 1.25, "Spike count > 0.3")
    p2 = Panel(650, 70, 360, 360, -0.6, len(rows) - 0.4, 0, max(r["max_spike"] for r in rows) * 1.15, "Max spike")
    els = [f'<text x="{w/2}" y="30" text-anchor="middle" font-size="24" font-weight="700">Tear-level dial: no divergence, rare severe spikes at full tear</text>']
    els += p1.axes(xticks=len(rows)-1, xlabel="run") + p1.bars(xs, [r["n_spikes"] for r in rows], COLORS["blue"], width=44)
    els += p2.axes(xticks=len(rows)-1, xlabel="run") + p2.bars(xs, [r["max_spike"] for r in rows], COLORS["red"], width=44)
    for p in (p1, p2):
        for i, label in enumerate(labels):
            safe = esc(label).replace("\n", " ")
            els.append(f'<text x="{p.sx(i):.1f}" y="{p.y + p.h + 78}" text-anchor="middle" font-size="12" fill="#333">{safe}</text>')
    write_svg("fig_tear_level_dial.svg", w, h, els)


def fig_geom_directionality():
    data = load_json("geom_olmoe.json")
    rows = data.get("layers") or []
    parsed = []
    for r in rows:
        nt_rows = r.get("normal_vs_tangent", [])
        if not nt_rows:
            continue
        nt = min(nt_rows, key=lambda e: abs(e.get("alpha", 0.0) - 2.0))
        normal = nt["normal"]
        tangent = nt["tangent"]
        parsed.append({
            "layer": r["layer_idx"],
            "dist": r["distance_to_tear"]["rel_median"],
            "normal_flip": normal["kk1_flip"],
            "tangent_flip": tangent["kk1_flip"],
            "normal_jump": normal["block_jump_rel"],
            "tangent_jump": tangent["block_jump_rel"],
            "normal_kl": normal["logit_kl_bs"],
            "tangent_kl": tangent["logit_kl_bs"],
        })
    if not parsed:
        print("skip fig_geom_directionality.svg: unrecognized geom_olmoe.json")
        return
    w, h = 1120, 660
    layers = [str(r["layer"]) for r in parsed]
    x = list(range(len(parsed)))
    p1 = Panel(90, 80, 280, 360, -0.5, len(parsed)-0.5, 0, 1.05, "k/k+1 flip rate")
    p2 = Panel(450, 80, 280, 360, -0.5, len(parsed)-0.5, 0, max(max(r["normal_jump"], r["tangent_jump"]) for r in parsed) * 1.25 + 1e-6, "Block jump")
    kl_vals = [v for r in parsed for v in (r["normal_kl"], r["tangent_kl"])]
    p3 = Panel(
        810,
        80,
        240,
        360,
        -0.5,
        len(parsed)-0.5,
        min(0.0, min(kl_vals) * 1.25),
        max(kl_vals) * 1.35 + 1e-6,
        "Downstream KL",
    )
    els = [f'<text x="{w/2}" y="32" text-anchor="middle" font-size="24" font-weight="700">Directional fragility: normal vs tangent perturbations</text>']
    for p, keyn, keyt, color in [
        (p1, "normal_flip", "tangent_flip", COLORS["blue"]),
        (p2, "normal_jump", "tangent_jump", COLORS["orange"]),
        (p3, "normal_kl", "tangent_kl", COLORS["purple"]),
    ]:
        els += p.axes(xticks=max(1, len(parsed)-1), xlabel="layer")
        els += p.bars([i - 0.12 for i in x], [r[keyn] for r in parsed], color, width=42, opacity=0.85)
        els += p.bars([i + 0.12 for i in x], [r[keyt] for r in parsed], COLORS["gray"], width=42, opacity=0.55)
        for i, lab in enumerate(layers):
            els.append(f'<text x="{p.sx(i):.1f}" y="{p.y + p.h + 75}" text-anchor="middle" font-size="13" fill="#333">L{esc(lab)}</text>')
    els.append('<text x="900" y="530" font-size="14" fill="#222">colored: normal</text>')
    els.append('<text x="900" y="552" font-size="14" fill="#555">gray: tangent</text>')
    dist_txt = ", ".join(f'L{r["layer"]}: {100*r["dist"]:.2f}%' for r in parsed)
    els.append(f'<text x="90" y="590" font-size="14" fill="#333">median distance-to-tear / ||h||: {esc(dist_txt)}</text>')
    write_svg("fig_geom_directionality.svg", w, h, els)


def fig_clamp_sweep():
    data = load_json("clamp_olmoe.json")
    rows = data.get("clamp_sweep") or []
    parsed = []
    for i, r in enumerate(rows):
        label = r.get("tag", f"c{i}")
        parsed.append({
            "label": label,
            "hardG": metric(r, "hardG"),
            "m2": metric(r, "M2_tear_median"),
            "cliff": metric(r, "cliff_abs_median"),
            "expert_norm": metric(r, "expert_norm_abs_median"),
            "jump": metric(r, "hardJump_abs"),
            "ppl": metric(r, "ppl"),
        })
    if not parsed:
        print("skip fig_clamp_sweep.svg: unrecognized clamp_olmoe.json")
        return
    w, h = 1180, 720
    x = list(range(len(parsed)))
    p1 = Panel(80, 80, 300, 350, -0.5, len(parsed)-0.5, 15.8, max(r["hardG"] for r in parsed) * 1.01 + 1e-6, "Topology: hardG")
    p2 = Panel(450, 80, 300, 350, -0.5, len(parsed)-0.5, 0, max(r["cliff"] for r in parsed) * 1.15 + 1e-6, "Absolute expert cliff")
    p3 = Panel(820, 80, 280, 350, -0.5, len(parsed)-0.5, 0, max(r["ppl"] for r in parsed) * 1.1 + 1e-6, "Hook-path ppl")
    els = [f'<text x="{w/2}" y="32" text-anchor="middle" font-size="24" font-weight="700">Clamp caps amplitude, not topology</text>']

    def categorical_axes(p, yticks=4):
        out = [
            f'<rect x="{p.x}" y="{p.y}" width="{p.w}" height="{p.h}" fill="#fff" stroke="#d0d0d0"/>',
            f'<text x="{p.x}" y="{p.y - 16}" font-size="20" font-weight="700" fill="#111">{esc(p.title)}</text>',
        ]
        for i in range(yticks + 1):
            yy = p.y + p.h * i / yticks
            val = p.ymax - (p.ymax - p.ymin) * i / yticks
            out.append(f'<line x1="{p.x}" y1="{yy:.1f}" x2="{p.x + p.w}" y2="{yy:.1f}" stroke="#eee"/>')
            out.append(f'<text x="{p.x - 10}" y="{yy + 5:.1f}" text-anchor="end" font-size="13" fill="#444">{val:.2f}</text>')
        return out

    for p, key, color in [(p1, "hardG", COLORS["purple"]), (p2, "cliff", COLORS["orange"]), (p3, "ppl", COLORS["red"])]:
        els += categorical_axes(p)
        vals = [r[key] for r in parsed]
        if key == "hardG":
            els.append(p.line(x, vals, color, width=2.8))
            els += p.points(x, vals, color, r=4.0)
        else:
            els += p.bars(x, vals, color, width=42, opacity=0.82)
        for i, r in enumerate(parsed):
            els.append(f'<text x="{p.sx(i):.1f}" y="{p.y + p.h + 72}" text-anchor="middle" font-size="12" fill="#333">{esc(r["label"])}</text>')
    els.append(f'<text x="80" y="620" font-size="13" fill="#555">native ppl={data["native_ppl"]:.2f}; hook hard baseline ppl={math.exp(data["hard_hook_base_nll"]):.2f}. Compare clamp rows within the hook path.</text>')
    els.append(f'<text x="80" y="646" font-size="13" fill="#555">abs jump: {parsed[0]["jump"]:.3f} → {parsed[-1]["jump"]:.3f}; M2 stays {parsed[0]["m2"]:.3f} → {parsed[-1]["m2"]:.3f}.</text>')
    write_svg("fig_clamp_sweep.svg", w, h, els)


def fig_cross_model_static():
    olmoe = load_json("sweep_olmoe.json")
    qwen = load_json("sweep_qwen.json")
    def stats(data):
        layers = [e for e in data["per_layer"] if "error" not in e]
        def mean(path):
            vals = []
            for e in layers:
                cur = e
                for key in path:
                    cur = cur.get(key) if isinstance(cur, dict) else None
                if cur is not None:
                    vals.append(cur)
            return sum(vals) / len(vals)
        return {
            "hardG": mean(["M3a_random", "hard_topk", "median"]),
            "m2": mean(["M2", "tear_median_near_boundary"]),
            "near": mean(["M1", "near_boundary_frac(<0.05)"]),
            "jump": mean(["M3a_jump_rel", "hard_topk", "median"]),
            "tied": mean(["M3a_random", "hard_tied(ctl)", "median"]),
        }
    rows = [("OLMoE", stats(olmoe)), ("Qwen-MoE", stats(qwen))]
    w, h = 1080, 640
    metrics = [
        ("hardG", "hardG", 0.0, 17.0, COLORS["purple"]),
        ("m2", "M2 cliff", 0.0, 0.8, COLORS["blue"]),
        ("near", "near-boundary", 0.0, 1.05, COLORS["green"]),
        ("jump", "block jump", 0.0, 0.45, COLORS["orange"]),
    ]
    els = [f'<text x="{w/2}" y="32" text-anchor="middle" font-size="24" font-weight="700">Tear replicates across released MoEs</text>']

    def categorical_axes(p, yticks=4):
        out = [
            f'<rect x="{p.x}" y="{p.y}" width="{p.w}" height="{p.h}" fill="#fff" stroke="#d0d0d0"/>',
            f'<text x="{p.x}" y="{p.y - 16}" font-size="20" font-weight="700" fill="#111">{esc(p.title)}</text>',
        ]
        for i in range(yticks + 1):
            yy = p.y + p.h * i / yticks
            val = p.ymax - (p.ymax - p.ymin) * i / yticks
            out.append(f'<line x1="{p.x}" y1="{yy:.1f}" x2="{p.x + p.w}" y2="{yy:.1f}" stroke="#eee"/>')
            out.append(f'<text x="{p.x - 10}" y="{yy + 5:.1f}" text-anchor="end" font-size="13" fill="#444">{val:.2f}</text>')
        return out

    panel_positions = [(80, 80), (580, 80), (80, 360), (580, 360)]
    for (key, title, ymin, ymax, color), (px, py) in zip(metrics, panel_positions):
        p = Panel(px, py, 380, 170, -0.5, 1.5, ymin, ymax, title)
        els += categorical_axes(p)
        vals = [s[key] for _, s in rows]
        els += p.bars([0, 1], vals, color, width=70, opacity=0.84)
        for i, ((model, _), val) in enumerate(zip(rows, vals)):
            yy = p.sy(val)
            els.append(f'<text x="{p.sx(i):.1f}" y="{yy-7:.1f}" text-anchor="middle" font-size="14" fill="#222">{val:.2f}</text>')
            els.append(f'<text x="{p.sx(i):.1f}" y="{p.y + p.h + 34}" text-anchor="middle" font-size="14" fill="#333">{esc(model)}</text>')
    els.append('<text x="80" y="610" font-size="13" fill="#555">Separate y-scales make the smaller M2 / prevalence / jump metrics readable; values are per-layer aggregates.</text>')
    write_svg("fig_cross_model_static.svg", w, h, els)


def fig_diffquotient_divergence():
    data = load_json("sweep_olmoe.json")
    layers = [e for e in data["per_layer"] if "error" not in e]
    xs = [e["layer_idx"] for e in layers]
    hard = [e["M3a_random"]["hard_topk"]["median"] for e in layers]
    soft = [e["M3a_random"]["soft_edge"]["median"] for e in layers]
    tied = [e["M3a_random"]["hard_tied(ctl)"]["median"] for e in layers]
    w, h = 1000, 560
    p = Panel(90, 80, 820, 360, min(xs), max(xs), 0.0, max(hard) * 1.12, "Max difference quotient ||dy||/||dx|| (8-path median)")
    els = [f'<text x="{w/2}" y="32" text-anchor="middle" font-size="23" font-weight="700">Difference-quotient divergence: hard top-k tears, soft/tied stay continuous</text>']
    els += p.axes(xticks=max(1, len(xs) - 1), xlabel="OLMoE layer")
    els.append(f'<line x1="{p.x}" y1="{p.sy(1.0):.1f}" x2="{p.x + p.w}" y2="{p.sy(1.0):.1f}" stroke="#bbb" stroke-dasharray="6 4"/>')
    els.append(f'<text x="{p.x + p.w - 6}" y="{p.sy(1.0) - 8:.1f}" text-anchor="end" font-size="13" fill="#999">continuous baseline = 1x</text>')
    els += [p.line(xs, hard, COLORS["purple"])] + p.points(xs, hard, COLORS["purple"])
    els += [p.line(xs, soft, COLORS["blue"])] + p.points(xs, soft, COLORS["blue"])
    els += [p.line(xs, tied, COLORS["gray"])] + p.points(xs, tied, COLORS["gray"])
    lx, ly = 690, 235
    for i, (lab, col) in enumerate([("hard top-k  (~16x)", COLORS["purple"]), ("soft edge  (~1x)", COLORS["blue"]), ("tied control  (~1x)", COLORS["gray"])]):
        els.append(f'<rect x="{lx}" y="{ly + i*22}" width="18" height="11" fill="{col}"/>')
        els.append(f'<text x="{lx + 26}" y="{ly + i*22 + 10}" font-size="14" fill="#222">{lab}</text>')
    els.append(f'<text x="90" y="500" font-size="13" fill="#555">Hard top-k diverges at the C0 discontinuity (~16x) at every layer; continuity-guaranteed controls stay ~1x.</text>')
    write_svg("fig_diffquotient_divergence.svg", w, h, els)


def fig_continuity_scaling():
    data = load_json("scaling_synth.json")
    res = data["resolutions"]
    lx = [math.log10(t) for t in res]
    pm = data["per_mode"]
    series = [
        ("hard_topk", "purple", "hard top-k"),
        ("soft_edge", "blue", "soft edge"),
        ("hard_tied(ctl)", "gray", "tied control"),
    ]
    ys = {m: [math.log10(v) for v in pm[m]["max_quotient_median"]] for m, _, _ in series}
    ymin = min(min(v) for v in ys.values()) - 0.3
    ymax = max(max(v) for v in ys.values()) + 0.3
    w, h = 1000, 560
    p = Panel(110, 80, 800, 360, min(lx), max(lx), ymin, ymax,
              "log10  max ||dy||/||dx||  (median over 48 synthetic boundary paths)")
    els = [f'<text x="{w/2}" y="32" text-anchor="middle" font-size="22" font-weight="700">Continuity-signature scaling: hard top-k diverges at exponent ~1 (order-0 jump)</text>']
    els += p.axes(xticks=4, yticks=4, xlabel="log10  grid resolution T")
    # slope-1 reference through the hard series' first point
    hx0, hy0 = lx[0], ys["hard_topk"][0]
    els.append(f'<line x1="{p.sx(hx0):.1f}" y1="{p.sy(hy0):.1f}" x2="{p.sx(lx[-1]):.1f}" y2="{p.sy(hy0 + (lx[-1]-hx0)):.1f}" stroke="#bbb" stroke-dasharray="6 4"/>')
    els.append(f'<text x="{p.sx(lx[-1]) - 6:.1f}" y="{p.sy(hy0 + (lx[-1]-hx0)) - 8:.1f}" text-anchor="end" font-size="12" fill="#999">slope = 1 reference</text>')
    for m, col, _ in series:
        els.append(p.line(lx, ys[m], COLORS[col]))
        els += p.points(lx, ys[m], COLORS[col])
    lxl, lyl = 150, 360
    for i, (m, col, lab) in enumerate(series):
        els.append(f'<rect x="{lxl}" y="{lyl + i*22}" width="18" height="11" fill="{COLORS[col]}"/>')
        els.append(f'<text x="{lxl + 26}" y="{lyl + i*22 + 10}" font-size="14" fill="#222">{lab}: exponent p = {pm[m]["slope"]:.2f}  (R² = {pm[m]["r2"]:.3f})</text>')
    els.append(f'<text x="110" y="500" font-size="13" fill="#555">Synthetic block with an exact known C0 jump: the hard difference quotient scales as T^1.00 (slope=1), so growth = max_q[T_max]/max_q[T_min] = T_max/T_min by construction; continuity controls are flat (p~0).</text>')
    write_svg("fig_continuity_scaling.svg", w, h, els)


def fig_ksweep():
    data = load_json("ksweep_summary.json")
    ks = [1, 2, 4, 8]
    layers = [str(l) for l in data["layers"]]
    xpos = {1: 0, 2: 1, 4: 2, 8: 3}
    cols = ["purple", "blue", "gray"]
    offs = [-0.07, 0.0, 0.07]
    w, h = 1000, 540
    p = Panel(110, 80, 800, 360, -0.35, 3.35, 14.8, 17.2,
              "hardG (refinement growth = max_q[8000]/max_q[500]), 64 paths/layer")
    els = [f'<text x="{w/2}" y="32" text-anchor="middle" font-size="22" font-weight="700">Same-model k-sweep: hardG pinned at the grid ratio 16 for every k (exponent ~1)</text>']
    els += p.axes(xticks=3, yticks=4)
    for xi, k in enumerate(ks):
        els.append(f'<text x="{p.sx(xpos[k]):.1f}" y="{p.y + p.h + 24:.1f}" text-anchor="middle" font-size="14" fill="#444">k={k}</text>')
    # grid-ratio reference
    els.append(f'<line x1="{p.x}" y1="{p.sy(16.0):.1f}" x2="{p.x + p.w}" y2="{p.sy(16.0):.1f}" stroke="#c0392b" stroke-dasharray="6 4" opacity="0.7"/>')
    els.append(f'<text x="{p.x + p.w - 6}" y="{p.sy(16.0) - 8:.1f}" text-anchor="end" font-size="13" fill="#c0392b">grid ratio 8000/500 = 16</text>')
    for li, lay in enumerate(layers):
        col = COLORS[cols[li]]
        for k in ks:
            r = data["per_k"][str(k)][lay]
            x = xpos[k] + offs[li]
            els.append(f'<line x1="{p.sx(x):.1f}" y1="{p.sy(r["hardG_min"]):.1f}" x2="{p.sx(x):.1f}" y2="{p.sy(r["hardG_max"]):.1f}" stroke="{col}" stroke-width="1.4" opacity="0.55"/>')
            els.append(f'<circle cx="{p.sx(x):.1f}" cy="{p.sy(r["hardG_median"]):.1f}" r="3.6" fill="{col}"/>')
    ly = 110
    for li, lay in enumerate(layers):
        els.append(f'<rect x="150" y="{ly + li*22}" width="18" height="11" fill="{COLORS[cols[li]]}"/>')
        els.append(f'<text x="176" y="{ly + li*22 + 10}" font-size="14" fill="#222">layer {lay}  (median ●, 64-path min–max whisker)</text>')
    els.append(f'<text x="110" y="480" font-size="13" fill="#555">hardG median stays 15.92–16.03 across k=1,2,4,8 (exponent 0.998–1.001): the value is the protocol grid ratio, not a k-function. Soft/tied controls sit at ~1 (off-scale).</text>')
    write_svg("fig_ksweep.svg", w, h, els)


def main():
    fig_diffquotient_divergence()
    fig_continuity_scaling()
    fig_ksweep()
    fig_e64_training()
    fig_e64_loss_spikes()
    fig_tear_level_seed_bars()
    fig_geom_directionality()
    fig_clamp_sweep()
    fig_cross_model_static()


if __name__ == "__main__":
    main()
