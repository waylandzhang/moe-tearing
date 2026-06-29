"""
moe_tear_probe.py

Static "manifold tearing" probe for a REAL trained open MoE (e.g. OLMoE-1B-7B).

SCOPE (read this before running)
--------------------------------
This harness analyses RELEASED WEIGHTS only. It measures the *geometry* of the
trained router + experts. It validates / falsifies the part of the theory that
lives in the final model:

  (M1) Boundary prevalence : how often does the model route tokens near the REAL
        top-k boundary (small g_k - g_{k+1} margin)? For top-k routing the
        discontinuity is where the k-th and (k+1)-th experts swap -- NOT top1/top2
        (top1..topk are all already inside the active set).
  (M2) Tear magnitude      : at the k/k+1 swap, how big is the cliff
        ||E_{k-th}(h) - E_{(k+1)-th}(h)|| (normalised)? i.e. if the router
        flipped that one slot, how much would the block output jump? Large =>
        tears are real in the trained model; tiny => the trained router
        self-aligned the seams (itself a finding, not a null result).
  (M3) Continuity signature: port the validated toy test to the real block --
        interpolate a hidden state across a real routing boundary, refine the
        grid, and check whether ||dy||/||dx|| DIVERGES under the model's own
        hard routing while a soft-edged reimplementation stays bounded.
  (M4) (optional) expert specialisation by domain.

OUT OF SCOPE (needs training-time traces / a fresh instrumented training run on
a GPU cluster -- deliberately deferred):
  * loss-spike <-> routing-churn correlation
  * whether soft-edged routing would have prevented spikes during training

Released checkpoints ship weights, NOT per-step loss + routing traces, so the
spike-causation study cannot be done here. That is a future GPU experiment.

USAGE
-----
  # local logic check, NO model download (builds a mock MoE block):
  python3 moe_tear_probe.py --selftest
  python3 moe_tear_probe.py --layer1-batch

  # real run (on a GPU box, downloads ~14GB the first time):
  python3 moe_tear_probe.py --model allenai/OLMoE-1B-7B-0924 --device cuda
  python3 moe_tear_probe.py --model allenai/OLMoE-1B-7B-0924 --device mps   # Apple

  # section-6 inference robustness + zero-retrain re-gating quality:
  python3 moe_tear_probe.py --section6 --model allenai/OLMoE-1B-7B-0924 --device cuda --tau 0.02
"""

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Block-level reimplementations (work on ANY block exposing .gate + .experts)  #
# We re-derive routing ourselves so we can compare hard vs soft-edged on the   #
# SAME trained experts, and so the harness is architecture-agnostic.           #
#                                                                              #
# Two expert layouts are supported, normalised by the _moe_* adapters below:   #
#   * transformers <=4.x : .gate = nn.Linear, .experts = ModuleList[callable]  #
#   * transformers >=5.x : .gate = a *Router* module (returns a tuple) whose   #
#       .weight is (E,H); .experts = a *fused* module storing all experts as   #
#       3D tensors (e.g. OlmoeExperts: gate_up_proj (E,2*I,H), down_proj       #
#       (E,H,I), act_fn). We replicate the module's OWN single-expert forward  #
#       from those tensors -- no hand-guessed FFN math.                        #
# --------------------------------------------------------------------------- #

def _moe_num_experts(block):
    exp = block.experts
    if hasattr(exp, "num_experts"):           # fused (5.x)
        return int(exp.num_experts)
    return len(exp)                           # ModuleList (4.x)


def _gate_logits(block, h):
    """Raw router logits (T, E), regardless of gate type."""
    g = block.gate
    if isinstance(g, nn.Linear):              # 4.x plain-Linear router
        return g(h)
    if hasattr(g, "weight"):                  # 5.x Router module: logits = F.linear(h, weight)
        return F.linear(h, g.weight)
    raise RuntimeError(f"unknown gate type: {type(g).__name__}")


# Experiment knob: when set, clamp the expert's SwiGLU intermediate (fused path) or
# output (ModuleList path) to [-c, c]. Acts AFTER routing, so the top-k decision is
# never affected -- this is the DeepSeek "SwiGLU clamping" decomposition: cap the
# magnitude of what a swapped expert contributes WITHOUT touching the discontinuity.
_EXPERT_CLAMP = None

# Whether to renormalize the gathered top-k router weights (norm_topk_prob).
# The released OLMoE-1B-7B-0924 and Qwen1.5-MoE-A2.7B configs both ship
# norm_topk_prob=false: the model gathers full-softmax weights at the top-k
# slots WITHOUT renormalizing. set_norm_topk() honors that per model config.
_NORM_TOPK = True


def set_expert_clamp(c):
    global _EXPERT_CLAMP
    _EXPERT_CLAMP = c


def set_norm_topk(model, override="auto"):
    """Set the global renorm convention from the model config (override in
    {auto,on,off}; auto reads model.config.norm_topk_prob)."""
    global _NORM_TOPK
    cfg = bool(getattr(model.config, "norm_topk_prob", True))
    _NORM_TOPK = cfg if override == "auto" else (override == "on")
    print(f"[gating] norm_topk_prob config={cfg} override={override} -> renorm={_NORM_TOPK}")
    return _NORM_TOPK


def _expert_forward(block, e, h):
    """Output of expert e on h: [N, H] -> [N, H]."""
    exp = block.experts
    if isinstance(exp, nn.ModuleList):        # 4.x callable expert modules
        out = exp[e](h)
        if _EXPERT_CLAMP is not None:         # no intermediate exposed -> clamp output
            out = out.clamp(-_EXPERT_CLAMP, _EXPERT_CLAMP)
        return out
    if hasattr(exp, "gate_up_proj") and hasattr(exp, "down_proj"):  # 5.x fused (OLMoE-style SwiGLU)
        gate, up = F.linear(h, exp.gate_up_proj[e]).chunk(2, dim=-1)
        inter = exp.act_fn(gate) * up
        if _EXPERT_CLAMP is not None:         # DeepSeek-style SwiGLU-intermediate clamp
            inter = inter.clamp(-_EXPERT_CLAMP, _EXPERT_CLAMP)
        return F.linear(inter, exp.down_proj[e])
    raise RuntimeError(f"unknown experts layout: {type(exp).__name__}")


def block_gates(block, h):
    """Return (gates, logits). h: [T, H] -> gates: [T, E]."""
    logits = _gate_logits(block, h)
    return F.softmax(logits, dim=-1), logits


def _combine(block, h, weights):
    """Weighted sum of expert outputs; computes only experts with nonzero
    weight for the tokens that use them. weights: [T, E]. -> [T, H]."""
    out = torch.zeros_like(h)
    E = weights.shape[-1]
    for e in range(E):
        col = weights[:, e]
        mask = col > 0
        if mask.any():
            out[mask] += col[mask].unsqueeze(-1) * _expert_forward(block, e, h[mask])
    return out


def forward_hard(block, h, k):
    gates, _ = block_gates(block, h)
    topv, topi = gates.topk(k, dim=-1)
    if _NORM_TOPK:
        topv = topv / topv.sum(dim=-1, keepdim=True)      # norm_topk_prob (per model config)
    w = torch.zeros_like(gates)
    w.scatter_(1, topi, topv)
    return _combine(block, h, w), (w > 0)


def forward_hard_tied(block, h, k):
    """NEGATIVE CONTROL: route exactly like forward_hard, but force every expert
    to be experts[0]. Switching experts then CANNOT change the output, so the map
    must be continuous (growth ~1x). If this control ever tears, the 'tear' is an
    artefact of the routing/weight arithmetic, not of expert disagreement."""
    gates, _ = block_gates(block, h)
    topv, _ = gates.topk(k, dim=-1)
    if _NORM_TOPK:
        wsum = (topv / topv.sum(dim=-1, keepdim=True)).sum(dim=-1, keepdim=True)  # == 1
    else:
        wsum = topv.sum(dim=-1, keepdim=True)             # native: gathered top-k mass (< 1)
    return wsum * _expert_forward(block, 0, h), None, None


def forward_soft_edge(block, h, tau, delta=0.05):
    """Partition-of-unity gate with a CONTINUITY GUARANTEE (no hard fallback).

    tau_eff = min(tau, (1-delta)*gmax) keeps the top expert's weight
    >= delta*gmax > 0, so the active set is never empty and the map is continuous
    everywhere (an expert enters/leaves the active set at weight exactly 0). The
    earlier hard top-1 fallback is GONE -- it was itself a discontinuity.

    Returns (out, active_mask, empty_plain) where empty_plain flags tokens for
    which the *plain* relu(g - tau) gate WOULD have emptied (g_max < tau). It is a
    diagnostic only: report its rate so the continuity claim is not free."""
    gates, _ = block_gates(block, h)
    gmax = gates.max(dim=-1, keepdim=True).values
    tau_eff = torch.minimum(torch.full_like(gmax, tau), (1.0 - delta) * gmax)
    w = F.relu(gates - tau_eff)
    w = w / w.sum(dim=-1, keepdim=True)
    empty_plain = gmax.squeeze(-1) < tau
    return _combine(block, h, w), (w > 0), empty_plain


# --------------------------------------------------------------------------- #
# Deep-Manifold geometry: raw-logit boundary normal, distance-to-tear, swap    #
# --------------------------------------------------------------------------- #

def _gate_weight_rows(block):
    """Raw router weight matrix W: (E, H). Row e is the logit direction of expert e."""
    g = block.gate
    if isinstance(g, nn.Linear):
        return g.weight
    if hasattr(g, "weight"):
        return g.weight
    raise RuntimeError(f"unknown gate type: {type(g).__name__}")


def boundary_geometry(block, h, k):
    """Raw-logit k/k+1 boundary geometry per token (NOT softmax margin).

    The active-set boundary is {x : logit_k(x) = logit_{k+1}(x)}, a hyperplane
    whose normal in hidden space is n = W_k - W_{k+1}. Returns the per-token
    normal, the signed distance-to-tear (logit_k - logit_{k+1}) / ||W_k - W_{k+1}||,
    and the expert pair that swaps across it."""
    logits = _gate_logits(block, h)                       # (T, E)
    E = logits.shape[-1]
    kk = min(k, E - 1)
    topv, topi = logits.topk(kk + 1, dim=-1)
    ek = topi[:, kk - 1]                                  # k-th: last INSIDE
    ek1 = topi[:, kk]                                     # (k+1)-th: first OUT
    W = _gate_weight_rows(block)                          # (E, H)
    n = W[ek] - W[ek1]                                    # (T, H) raw-logit normal
    n_norm = n.norm(dim=-1)                               # (T,)
    logit_margin = topv[:, kk - 1] - topv[:, kk]          # (T,) = n . h  (>= 0)
    distance = logit_margin / n_norm.clamp_min(1e-12)     # (T,) hidden-space distance
    n_hat = n / n_norm.clamp_min(1e-12).unsqueeze(-1)
    return {"ek": ek, "ek1": ek1, "n_hat": n_hat, "distance": distance,
            "logit_margin": logit_margin, "n_norm": n_norm}


def _orthogonal_tangent(n_hat, gen):
    """Per-token random unit vector orthogonal to n_hat (in hidden space)."""
    v = torch.randn(n_hat.shape, generator=gen, dtype=n_hat.dtype, device=n_hat.device)
    v = v - (v * n_hat).sum(-1, keepdim=True) * n_hat     # remove component along n_hat
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def geom_perturb(block, h, k, alpha, direction, gen=None):
    """Perturb each token by alpha * distance_to_tear along the boundary normal
    (toward the boundary) or a tangent of the SAME magnitude. Reports whether the
    k/k+1 pair flipped and whether the top-k set changed at all."""
    geo = boundary_geometry(block, h, k)
    step = (alpha * geo["distance"]).unsqueeze(-1)        # (T, 1)
    if direction == "normal":
        # margin >= 0, so move toward the boundary: new margin = (1 - alpha)*margin,
        # which crosses (flips k/k+1) once alpha > 1.
        h_pert = h - step * geo["n_hat"]
    elif direction == "tangent":
        h_pert = h + step * _orthogonal_tangent(geo["n_hat"], gen)
    else:
        raise ValueError(f"unknown direction: {direction}")
    E = _gate_logits(block, h).shape[-1]
    kk = min(k, E - 1)
    old_mask = _topk_set_masks(_gate_logits(block, h), kk)
    new_mask = _topk_set_masks(_gate_logits(block, h_pert), kk)
    ek1_in = new_mask.gather(1, geo["ek1"].unsqueeze(1)).squeeze(1)   # (T,) bool
    return h_pert, {
        "kk1_flip": ek1_in,                              # the just-outside expert entered top-k
        "any_topk_flip": (new_mask != old_mask).any(-1),
    }


def forward_swap_kk1(block, h, k):
    """COUNTERFACTUAL: keep the hard top-k weights, but route the k-th slot to the
    (k+1)-th expert instead of the k-th. Isolates whether the local expert cliff
    transmits downstream, with NO perturbation (so no eps=0 hook artefact)."""
    gates, _ = block_gates(block, h)
    E = gates.shape[-1]
    kk = min(k, E - 1)
    topv, topi = gates.topk(kk, dim=-1)
    topv = topv / topv.sum(dim=-1, keepdim=True)         # norm_topk_prob
    w = torch.zeros_like(gates)
    w.scatter_(1, topi, topv)
    full = gates.topk(kk + 1, dim=-1).indices
    ek, ek1 = full[:, kk - 1], full[:, kk]
    rows = torch.arange(w.shape[0], device=w.device)
    wk = w[rows, ek].clone()                              # weight on the k-th expert
    w[rows, ek] = 0.0
    w[rows, ek1] = w[rows, ek1] + wk                      # move it onto the (k+1)-th
    return _combine(block, h, w)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def routing_stats(block, H, k):
    """(M1) boundary prevalence at the REAL top-k boundary (g_k - g_{k+1})."""
    gates, _ = block_gates(block, H)
    E = gates.shape[-1]
    kk = min(k, E - 1)
    topv, _ = gates.topk(kk + 1, dim=-1)
    margin = (topv[:, kk - 1] - topv[:, kk]).float()       # g_k - g_{k+1} (float: quantile/bf16-safe)
    return {
        "k_boundary": f"{kk}/{kk + 1}",
        "n_tokens": H.shape[0],
        "margin_median": margin.median().item(),
        "margin_p10": margin.quantile(0.10).item(),
        "near_boundary_frac(<0.05)": (margin < 0.05).float().mean().item(),
        "near_boundary_frac(<0.02)": (margin < 0.02).float().mean().item(),
    }


@torch.no_grad()
def tear_magnitude(block, H, k):
    """(M2) cliff at the k/k+1 swap: ||E_{k-th}(h) - E_{(k+1)-th}(h)|| normalised.
    For top-k routing this -- not top1/top2 -- is the expert pair that actually
    swaps across the active-set boundary."""
    gates, _ = block_gates(block, H)
    E = gates.shape[-1]
    kk = min(k, E - 1)
    topv, topi = gates.topk(kk + 1, dim=-1)
    ek = topi[:, kk - 1]                                   # k-th: last one INSIDE
    ek1 = topi[:, kk]                                      # (k+1)-th: first one OUT
    yk = torch.zeros_like(H)
    yk1 = torch.zeros_like(H)
    for e in range(E):
        m = ek == e
        if m.any():
            yk[m] = _expert_forward(block, e, H[m])
        m = ek1 == e
        if m.any():
            yk1[m] = _expert_forward(block, e, H[m])
    num = (yk - yk1).norm(dim=-1)
    den = yk.norm(dim=-1) + yk1.norm(dim=-1) + 1e-6
    tear = (num / den).float()                            # 0 = agree, ~1 = full cliff (bf16-safe)
    # (#2) cosine of the swapped pair: disentangles redundant (cos~1) from generic
    # non-redundant (cos~0, where tear~=sqrt2/2~=0.707) from specialised (cos<0).
    cos = ((yk * yk1).sum(dim=-1) / (yk.norm(dim=-1) * yk1.norm(dim=-1) + 1e-6)).float()
    margin = topv[:, kk - 1] - topv[:, kk]
    near = margin < 0.05
    cliff_abs = num.float()                              # ||E_k - E_{k+1}|| (absolute, NOT normalized)
    expert_norm_abs = ((yk.norm(dim=-1) + yk1.norm(dim=-1)) / 2).float()
    out = {
        "k_boundary": f"{kk}/{kk + 1}",
        "tear_median_all": round(tear.median().item(), 4),
        "tear_p90_all": round(tear.quantile(0.90).item(), 4),
        "near_boundary_frac(<0.05)": round(near.float().mean().item(), 4),
        "tear_median_near_boundary": (round(tear[near].median().item(), 4)
                                      if near.any() else float("nan")),
        # absolute-magnitude metrics (clamp acts on these; the normalized tear is scale-invariant)
        "cliff_abs_median": round(cliff_abs.median().item(), 4),
        "expert_norm_abs_median": round(expert_norm_abs.median().item(), 4),
        "cos_kk1_median": round(cos.median().item(), 4),
        "cos_kk1_p10": round(cos.quantile(0.10).item(), 4),
        "cos_kk1_p90": round(cos.quantile(0.90).item(), 4),
        "cos_kk1_neg_frac": round((cos < 0).float().mean().item(), 4),
    }
    return out


@torch.no_grad()
def soft_edge_diagnostics(block, H, tau):
    """Cost + continuity-guarantee diagnostics for the soft-edge gate."""
    _, mask, empty_plain = forward_soft_edge(block, H, tau)
    return {
        "avg_active_experts": mask.float().sum(-1).mean().item(),
        "empty_under_plain_relu_frac": empty_plain.float().mean().item(),
    }


def _summary(vals):
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "n": len(vals),
        "median": t.median().item(),
        "p95": t.quantile(0.95).item(),
        "min": t.min().item(),
        "max": t.max().item(),
    }


def _print_growth_summary(label, growths):
    print(f"\n[{label}]")
    for mode in ("hard_topk", "soft_edge", "hard_tied(ctl)"):
        s = _summary(growths[mode])
        print(f"  {mode:<16} n={s['n']:<3d} median={s['median']:.2f}x "
              f"p95={s['p95']:.2f}x min={s['min']:.2f}x max={s['max']:.2f}x")


def _topk_set_masks(gates, k):
    topi = gates.topk(k, dim=-1).indices
    masks = torch.zeros_like(gates, dtype=torch.bool)
    masks.scatter_(1, topi, True)
    return masks


@torch.no_grad()
def find_boundary_pair(block, H, k, mode="first", generator=None,
                       max_pool=2048, return_meta=False):
    """Pick a straight-path probe crossing a real top-k active-set boundary.

    mode="random" samples an unbiased boundary-crossing pair from the hidden-state
    pool. mode="targeted" chooses the pair with the largest top-k set difference,
    tie-breaking toward lower endpoint k/k+1 margins. Random is the main
    diagnostic; targeted is a pressure test / mechanism display.
    """
    gates, _ = block_gates(block, H)
    E = gates.shape[-1]
    kk = min(k, E - 1)
    if H.shape[0] > max_pool:
        idx = torch.linspace(0, H.shape[0] - 1, max_pool, device=H.device).long()
        H = H[idx]
        gates = gates[idx]
    masks = _topk_set_masks(gates, kk)
    topv, _ = gates.topk(kk + 1, dim=-1)
    margin = topv[:, kk - 1] - topv[:, kk]
    n = H.shape[0]

    def pack(a, b, pick_mode):
        set_diff = (masks[a] != masks[b]).sum().item()
        meta = {
            "mode": pick_mode,
            "k_boundary": f"{kk}/{kk + 1}",
            "set_diff": int(set_diff),
            "endpoint_margin_sum": (margin[a] + margin[b]).item(),
        }
        if return_meta:
            return H[a], H[b], meta
        return H[a], H[b]

    if n < 2:
        raise ValueError("Need at least two hidden states to probe a boundary.")

    if mode == "random":
        for _ in range(max(1000, 10 * n)):
            a = int(torch.randint(n, (1,), generator=generator).item())
            b = int(torch.randint(n, (1,), generator=generator).item())
            if a != b and (masks[a] != masks[b]).any():
                return pack(a, b, "random")

    if mode == "targeted":
        best = None
        best_score = None
        for a in range(n):
            diff = (masks[a].unsqueeze(0) != masks).sum(dim=1)
            diff[a] = -1
            max_diff = diff.max().item()
            if max_diff <= 0:
                continue
            candidates = torch.nonzero(diff == max_diff, as_tuple=False).squeeze(-1)
            j = candidates[(margin[candidates] + margin[a]).argmin()].item()
            score = (int(max_diff), -(margin[a] + margin[j]).item())
            if best_score is None or score > best_score:
                best_score = score
                best = (a, j)
        if best is not None:
            return pack(best[0], best[1], "targeted")

    # Deterministic fallback: first active-set change in the pool.
    a = 0
    for b in range(1, n):
        if (masks[b] != masks[a]).any():
            return pack(a, b, "first")
    return pack(0, n - 1, "no-boundary-fallback")


@torch.no_grad()
def continuity_signature(block, h_a, h_b, resolutions=(500, 2000, 8000),
                         k=2, tau=0.15):
    """(M3) max ||dy||/||dx|| as the path is refined. Returns per mode:
      growth   : max_quotient[-1]/max_quotient[0]; ~1x => continuous, ~refine => tear.
      jump_abs : (#1) max ||dy|| at the FINEST grid -- the absolute size of the
                 biggest single-step BLOCK-OUTPUT change. For a continuous map this
                 -> 0 as the grid refines; at a real jump it stays at the jump size.
      jump_rel : jump_abs / median ||y|| -- the jump as a fraction of typical output
                 norm. This is the honest 'how big does the WHOLE block output jump'
                 number (distinct from M2's per-slot cliff ||E_k - E_{k+1}||)."""
    modes = (
        ("hard_topk", lambda h: forward_hard(block, h, k)),
        ("soft_edge", lambda h: forward_soft_edge(block, h, tau)),
        ("hard_tied(ctl)", lambda h: forward_hard_tied(block, h, k)),  # neg control
    )
    res = {}
    for name, fn in modes:
        maxes = []
        jump_abs = jump_rel = float("nan")
        for T in resolutions:
            t = torch.linspace(0, 1, T, device=h_a.device,
                               dtype=h_a.dtype).unsqueeze(-1)
            path = h_a.unsqueeze(0) + t * (h_b - h_a).unsqueeze(0)   # [T,H]
            y, *_ = fn(path)
            dx = (path[1:] - path[:-1]).norm(dim=-1)
            dy = (y[1:] - y[:-1]).norm(dim=-1)
            maxes.append((dy / dx).max().item())
            jump_abs = dy.max().item()                              # finest grid wins (last iter)
            jump_rel = jump_abs / (y.norm(dim=-1).median().item() + 1e-6)
        growth = maxes[-1] / maxes[0] if maxes[0] > 0 else float("nan")
        res[name] = {"max_quotient": maxes, "growth": growth,
                     "jump_abs": jump_abs, "jump_rel": jump_rel}
    return res


def print_continuity_signature(sig, meta=None, indent="     "):
    if meta:
        print(f"{indent}path={meta['mode']} boundary={meta['k_boundary']} "
              f"set_diff={meta['set_diff']} "
              f"endpoint_margin_sum={meta['endpoint_margin_sum']:.4f}")
    for mode, d in sig.items():
        print(f"{indent}{mode:<16} max_quotient={['%.1f'%m for m in d['max_quotient']]} "
              f"growth={d['growth']:.1f}x  jump_rel={d['jump_rel']:.3f} (||Δy||/||y||)")


# --------------------------------------------------------------------------- #
# Model plumbing                                                               #
# --------------------------------------------------------------------------- #

def discover_moe_blocks(model):
    """Architecture-agnostic: any submodule exposing .gate + .experts."""
    blocks = []
    for name, mod in model.named_modules():
        if hasattr(mod, "gate") and hasattr(mod, "experts"):
            try:
                n_exp = _moe_num_experts(mod)
            except (TypeError, AttributeError):
                continue
            blocks.append((name, mod, n_exp))
    return blocks


def capture_hidden_states(model, tokenizer, texts, block, device):
    """Run the model, hook `block` to grab its INPUT hidden states (the pre-router
    h that the experts also consume). Returns [N_tokens, hidden]."""
    grabbed = {}

    def hook(_mod, inp, _out):
        grabbed["h"] = inp[0].detach()

    handle = block.register_forward_hook(hook)
    chunks = []
    try:
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True,
                            max_length=128).to(device)
            with torch.no_grad():
                model(**enc)
            h = grabbed["h"]
            chunks.append(h.reshape(-1, h.shape[-1]))
    finally:
        handle.remove()
    return torch.cat(chunks, dim=0)


def capture_hidden_states_multi(model, tokenizer, texts, blocks_by_name, device):
    """Hook several blocks at once: ONE model pass per text grabs every block's
    INPUT hidden states. Lets a full-model sweep capture all layers without
    re-running the model per layer. blocks_by_name: {name: module}.
    Returns {name: [N_tokens, hidden]} in the model's current dtype."""
    grabbed = {name: [] for name in blocks_by_name}

    def mk(name):
        def hook(_mod, inp, _out):
            h = inp[0].detach()
            grabbed[name].append(h.reshape(-1, h.shape[-1]))
        return hook

    handles = [mod.register_forward_hook(mk(name))
               for name, mod in blocks_by_name.items()]
    try:
        for txt in texts:
            enc = tokenizer(txt, return_tensors="pt", truncation=True,
                            max_length=128).to(device)
            with torch.no_grad():
                model(**enc)
    finally:
        for h in handles:
            h.remove()
    return {name: torch.cat(chunks, dim=0) for name, chunks in grabbed.items()}


def run_real(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    set_norm_topk(model, getattr(args, "norm_topk", "auto"))
    model.eval()
    k = args.k or getattr(model.config, "num_experts_per_tok", 2)

    blocks = discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("No MoE blocks (.gate+.experts) found in this model.")
    print(f"Found {len(blocks)} MoE blocks; routing k={k}. Probing layer {args.layer}.")
    name, block, n_exp = blocks[args.layer]
    print(f"  block='{name}', n_experts={n_exp}")

    texts = DEFAULT_PROBE_TEXTS[: (args.n_texts or 8)]
    H = capture_hidden_states(model, tok, texts, block, device).float()
    block_f = block.float()                               # metrics in fp32

    print(f"\n[M1] routing boundary prevalence (boundary = g_k vs g_(k+1)):")
    for key, v in routing_stats(block_f, H, k).items():
        print(f"     {key:<28} {v}")
    print(f"\n[M2] tear magnitude at the k/k+1 swap (0=agree, ~1=full cliff):")
    for key, v in tear_magnitude(block_f, H, k).items():
        print(f"     {key:<28} {v}")
    print(f"\n[soft-edge diagnostics] continuity-guaranteed gate (no hard fallback):")
    for key, v in soft_edge_diagnostics(block_f, H, args.tau).items():
        print(f"     {key:<28} {v:.4f}")
    print(f"\n[M3a] continuity signature over {args.m3_paths} RANDOM top-k boundary "
          f"paths (population diagnostic, growth median/p95):")
    gen = torch.Generator().manual_seed(args.path_seed)
    growths = {"hard_topk": [], "soft_edge": [], "hard_tied(ctl)": []}
    for _ in range(args.m3_paths):
        h_a, h_b, _ = find_boundary_pair(block_f, H, k, mode="random",
                                         generator=gen, return_meta=True)
        sig = continuity_signature(block_f, h_a, h_b, k=k, tau=args.tau)
        for mode in growths:
            growths[mode].append(sig[mode]["growth"])
    _print_growth_summary(f"M3a random paths (n={args.m3_paths}, k={k})", growths)

    print("\n[M3b] continuity signature on the TARGETED (worst-case) boundary path "
          "(pressure test, single deterministic pick):")
    h_a, h_b, meta = find_boundary_pair(block_f, H, k, mode="targeted", return_meta=True)
    sig = continuity_signature(block_f, h_a, h_b, k=k, tau=args.tau)
    print_continuity_signature(sig, meta)
    print("\nReading:")
    print("  hard_topk growth ~= refinement factor => tear in the trained model")
    print("  soft_edge growth ~1x                  => seam made continuous (same experts)")
    print("  hard_tied(ctl) growth ~1x             => NEG CONTROL ok (tear needs expert disagreement)")
    print("  M2 small + M1 low?  => trained router self-aligned seams (a finding, not a null);")
    print("                          the tear likely lives in TRAINING dynamics, see note 7B.")
    print("  empty_under_plain_relu_frac high?     => the continuity guarantee is doing real work;")
    print("                          report it so 'continuous' is not a free lunch.")


# --------------------------------------------------------------------------- #
# Layer-2 sweep: one-shot driver. Load the model ONCE, capture every MoE block's #
# hidden states in a single pass, run M1/M2/M3 on each block, dump JSON. Built   #
# so a GPU box can run-then-shutdown and we analyse the JSON offline.            #
# --------------------------------------------------------------------------- #

def _json_safe(o):
    """Recursively replace NaN/Inf with None so the dump is valid JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    return o


def _fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "nan"
    return f"{v:.4f}"


def _print_sweep_summary(out):
    print("\n" + "=" * 86)
    print(f"SWEEP SUMMARY  model={out['model']}  k={out['k']}  tau={out['tau']}  "
          f"n_texts={out['n_texts']}")
    print("=" * 86)
    print("M2_tear_near -> 1 = real trained-model seam tear;  -> 0 = router self-aligned the seam.")
    print("cos_kk1: ~1 redundant experts, ~0 non-redundant (tear ~= .71 baseline), <0 specialised.")
    print("hardJump = whole-block ||Δy||/||y|| at finest grid (the ACTUAL output jump, vs M2's per-slot cliff).")
    print("hardGrowth ~= grid factor => discontinuity exists;  softGrowth_p95 ~1 => continuous.")
    print("-" * 100)
    hdr = (f"{'L':>3} {'n_exp':>5} {'M2_near':>8} {'cos_kk1':>8} {'M1<.05':>7} "
           f"{'hardG_med':>10} {'hardJump':>9} {'softJump':>9} {'softG_p95':>10} {'tiedG_p95':>10}")
    print(hdr)
    for e in out["per_layer"]:
        if "error" in e:
            print(f"{e['layer_idx']:>3}  ERROR: {e['error'][:64]}")
            continue
        m1, m2, m3 = e["M1"], e["M2"], e["M3a_random"]
        jr = e.get("M3a_jump_rel", {})
        hj = jr.get("hard_topk", {}).get("median", float("nan"))
        sj = jr.get("soft_edge", {}).get("median", float("nan"))
        print(f"{e['layer_idx']:>3} {e['n_experts']:>5} "
              f"{_fmt(m2.get('tear_median_near_boundary')):>8} "
              f"{_fmt(m2.get('cos_kk1_median')):>8} "
              f"{m1['near_boundary_frac(<0.05)']:>7.3f} "
              f"{m3['hard_topk']['median']:>9.1f}x "
              f"{_fmt(hj):>9} {_fmt(sj):>9} "
              f"{m3['soft_edge']['p95']:>9.2f}x "
              f"{m3['hard_tied(ctl)']['p95']:>9.2f}x")


def run_sweep(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    set_norm_topk(model, getattr(args, "norm_topk", "auto"))
    model.eval()
    k = args.k or getattr(model.config, "num_experts_per_tok", 2)

    blocks = discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("No MoE blocks (.gate+.experts) found in this model.")
    n_blocks = len(blocks)

    if args.sweep_layers:
        idxs = [int(x) for x in args.sweep_layers.split(",") if x.strip() != ""]
        idxs = [i for i in idxs if 0 <= i < n_blocks]
    else:
        idxs = list(range(n_blocks))
    target = [(i, blocks[i][0], blocks[i][2]) for i in idxs]
    print(f"Found {n_blocks} MoE blocks; routing k={k}. Sweeping {len(target)} "
          f"layer(s): {idxs}")

    nt = args.n_texts or len(SWEEP_PROBE_TEXTS)
    texts = SWEEP_PROBE_TEXTS[:nt]
    # Capture BEFORE any fp32 conversion (model still in bf16 -> cheap memory).
    target_blocks = {name: blocks[i][1] for (i, name, _) in target}
    print(f"Capturing hidden states from {len(target_blocks)} blocks over "
          f"{len(texts)} texts ...")
    states = capture_hidden_states_multi(model, tok, texts, target_blocks, device)

    results = []
    for (i, name, n_exp) in target:
        blk = blocks[i][1]
        H = states[name].float()
        try:
            blk.float()                                   # fp32 metrics for THIS block
            entry = {
                "layer_idx": i, "block_name": name, "n_experts": n_exp,
                "n_tokens": int(H.shape[0]),
                "M1": routing_stats(blk, H, k),
                "M2": tear_magnitude(blk, H, k),
                "soft": soft_edge_diagnostics(blk, H, args.tau),
            }
            gen = torch.Generator().manual_seed(args.path_seed)
            growths = {"hard_topk": [], "soft_edge": [], "hard_tied(ctl)": []}
            jumps = {"hard_topk": [], "soft_edge": [], "hard_tied(ctl)": []}
            for _ in range(args.m3_paths):
                h_a, h_b, _ = find_boundary_pair(blk, H, k, mode="random",
                                                 generator=gen, return_meta=True)
                sig = continuity_signature(blk, h_a, h_b, k=k, tau=args.tau)
                for m in growths:
                    growths[m].append(sig[m]["growth"])
                    jumps[m].append(sig[m]["jump_rel"])          # (#1) whole-block jump / ||y||
            entry["M3a_random"] = {m: _summary(v) for m, v in growths.items()}
            entry["M3a_jump_rel"] = {m: _summary(v) for m, v in jumps.items()}

            h_a, h_b, meta = find_boundary_pair(blk, H, k, mode="targeted",
                                                return_meta=True)
            sig = continuity_signature(blk, h_a, h_b, k=k, tau=args.tau)
            entry["M3b_targeted"] = {"meta": meta,
                                     **{m: sig[m]["growth"] for m in sig}}
            results.append(entry)
            print(f"  [L{i:>3} {name}] M2_near="
                  f"{_fmt(entry['M2'].get('tear_median_near_boundary'))} "
                  f"M1<.05={entry['M1']['near_boundary_frac(<0.05)']:.3f} "
                  f"hardG={entry['M3a_random']['hard_topk']['median']:.1f}x "
                  f"softG_p95={entry['M3a_random']['soft_edge']['p95']:.2f}x")
        except Exception as e:                            # one bad layer must not kill the sweep
            print(f"  [L{i} {name}] FAILED: {e}")
            results.append({"layer_idx": i, "block_name": name, "error": str(e)})
        finally:
            blk.to(dtype)                                 # restore bf16 -> release fp32 memory

    out = {
        "model": args.model, "device": device, "dtype": str(dtype), "k": k,
        "tau": args.tau, "n_texts": len(texts), "n_blocks_total": n_blocks,
        "m3_paths": args.m3_paths, "path_seed": args.path_seed,
        "torch": torch.__version__, "per_layer": results,
    }
    path = args.out or f"sweep_{args.model.split('/')[-1]}.json"
    with open(path, "w") as f:
        json.dump(_json_safe(out), f, indent=2, ensure_ascii=False)
    _print_sweep_summary(out)
    print(f"\nWrote {path}  ({len(results)} layers). Safe to shut the GPU down now.")


# --------------------------------------------------------------------------- #
# Section-6 experiments: inference robustness + zero-retrain re-gating quality #
# --------------------------------------------------------------------------- #

def _replace_first_tensor(output, replacement):
    """Forward-hook helper: preserve auxiliary router outputs when present."""
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise RuntimeError(f"cannot replace MoE output type: {type(output).__name__}")


def _parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _encode_text(tokenizer, text, device, max_length):
    return tokenizer(text, return_tensors="pt", truncation=True,
                     max_length=max_length).to(device)


def _n_predict_tokens(enc):
    return max(int(enc["input_ids"].numel()) - 1, 1)


def _ppl(nll):
    return math.exp(min(float(nll), 20.0))


@torch.no_grad()
def _lm_eval(model, tokenizer, texts, device, max_length):
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        enc = _encode_text(tokenizer, text, device, max_length)
        out = model(**enc, labels=enc["input_ids"])
        n = _n_predict_tokens(enc)
        total_nll += float(out.loss) * n
        total_tokens += n
    nll = total_nll / max(total_tokens, 1)
    return {"nll": nll, "ppl": _ppl(nll), "n_tokens": total_tokens}


def _install_soft_regating_hooks(blocks, tau, stats=None):
    handles = []
    for _name, block, _n_exp in blocks:
        def hook(mod, inp, out, tau=tau, stats=stats):
            h = inp[0]
            flat = h.reshape(-1, h.shape[-1])
            y, mask, empty = forward_soft_edge(mod, flat, tau)
            if stats is not None:
                stats["active"].append(mask.float().sum(-1).mean().item())
                stats["empty"].append(empty.float().mean().item())
            return _replace_first_tensor(out, y.reshape_as(h).to(dtype=h.dtype))
        handles.append(block.register_forward_hook(hook))
    return handles


def _remove_hooks(handles):
    for handle in handles:
        handle.remove()


def run_regating_quality_loaded(args, model, tokenizer, blocks, device):
    texts = SWEEP_PROBE_TEXTS[: (args.n_texts or len(SWEEP_PROBE_TEXTS))]
    hard = _lm_eval(model, tokenizer, texts, device, args.max_length)
    stats = {"active": [], "empty": []}
    handles = _install_soft_regating_hooks(blocks, args.tau, stats)
    try:
        soft = _lm_eval(model, tokenizer, texts, device, args.max_length)
    finally:
        _remove_hooks(handles)
    out = {
        "mode": "soft_edge_regating_quality",
        "tau": args.tau,
        "n_texts": len(texts),
        "max_length": args.max_length,
        "hard": hard,
        "soft_edge": soft,
        "delta_nll": soft["nll"] - hard["nll"],
        "ppl_ratio": soft["ppl"] / max(hard["ppl"], 1e-12),
        "soft_avg_active_experts": (sum(stats["active"]) / len(stats["active"])
                                    if stats["active"] else None),
        "soft_empty_under_plain_relu_frac": (sum(stats["empty"]) / len(stats["empty"])
                                             if stats["empty"] else None),
    }
    print("\n[re-gating quality]")
    print(f"  hard       nll={hard['nll']:.4f} ppl={hard['ppl']:.2f} tokens={hard['n_tokens']}")
    print(f"  soft_edge  nll={soft['nll']:.4f} ppl={soft['ppl']:.2f} "
          f"delta={out['delta_nll']:.4f} ppl_ratio={out['ppl_ratio']:.3f}")
    print(f"  soft avg_active={_fmt(out['soft_avg_active_experts'])} "
          f"empty_plain={_fmt(out['soft_empty_under_plain_relu_frac'])}")
    return out


def _relative_noise(h, eps, seed):
    torch.manual_seed(seed)
    hf = h.float()
    noise = torch.randn(hf.shape, device=hf.device, dtype=torch.float32)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return noise * hf.norm(dim=-1, keepdim=True).clamp_min(1e-6) * eps


def _topk_flip_frac(block, h0, h1, k):
    g0, _ = block_gates(block, h0)
    g1, _ = block_gates(block, h1)
    m0 = _topk_set_masks(g0, min(k, g0.shape[-1] - 1))
    m1 = _topk_set_masks(g1, min(k, g1.shape[-1] - 1))
    return (m0 != m1).any(-1).float().mean().item()


def _logit_change(base_logits, pert_logits):
    b = base_logits[:, :-1, :].float()
    p = pert_logits[:, :-1, :].float()
    b_logp = F.log_softmax(b, dim=-1)
    p_logp = F.log_softmax(p, dim=-1)
    b_prob = b_logp.exp()
    kl = (b_prob * (b_logp - p_logp)).sum(-1).mean().item()
    top1 = (b.argmax(dim=-1) != p.argmax(dim=-1)).float().mean().item()
    return kl, top1


@torch.no_grad()
def _perturbed_forward(model, tokenizer, text, device, block, k, tau, eps,
                       mode, seed, max_length):
    enc = _encode_text(tokenizer, text, device, max_length)
    stats = {}

    def hook(mod, inp, out):
        h = inp[0]
        flat = h.reshape(-1, h.shape[-1])
        noise = _relative_noise(flat, eps, seed).to(dtype=flat.dtype)
        pert = flat + noise

        base_hard, _ = forward_hard(mod, flat, k)
        pert_hard, _ = forward_hard(mod, pert, k)
        base_soft, _, _ = forward_soft_edge(mod, flat, tau)
        pert_soft, _, _ = forward_soft_edge(mod, pert, tau)
        stats["topk_flip_frac"] = _topk_flip_frac(mod, flat, pert, k)
        stats["hard_block_jump_rel"] = (
            (pert_hard - base_hard).norm(dim=-1).median()
            / (base_hard.norm(dim=-1).median() + 1e-6)
        ).item()
        stats["soft_block_jump_rel"] = (
            (pert_soft - base_soft).norm(dim=-1).median()
            / (base_soft.norm(dim=-1).median() + 1e-6)
        ).item()

        y = pert_hard if mode == "hard" else pert_soft
        return _replace_first_tensor(out, y.reshape_as(h).to(dtype=h.dtype))

    handle = block.register_forward_hook(hook)
    try:
        out = model(**enc, labels=enc["input_ids"])
    finally:
        handle.remove()
    return out.loss.item(), out.logits.detach(), stats, _n_predict_tokens(enc)


@torch.no_grad()
def run_robustness_loaded(args, model, tokenizer, blocks, k, device):
    idx = min(max(args.layer, 0), len(blocks) - 1)
    name, block, n_exp = blocks[idx]
    texts = SWEEP_PROBE_TEXTS[: (args.n_texts or 8)]
    eps_values = _parse_float_list(args.perturb_eps)
    out = {
        "mode": "hidden_perturbation_robustness",
        "layer_idx": idx,
        "block_name": name,
        "n_experts": n_exp,
        "k": k,
        "tau": args.tau,
        "eps_values": eps_values,
        "n_texts": len(texts),
        "max_length": args.max_length,
        "per_eps": [],
    }
    print("\n[input/hidden perturbation robustness]")
    print(f"  probing layer={idx} n_experts={n_exp} k={k} tau={args.tau}")
    for eps in eps_values:
        rows = []
        for j, text in enumerate(texts):
            enc = _encode_text(tokenizer, text, device, args.max_length)
            base = model(**enc, labels=enc["input_ids"])
            base_loss = float(base.loss)
            base_logits = base.logits.detach()
            seed = args.perturb_seed + 1000 * j + int(round(eps * 1e9))
            hard_loss, hard_logits, stats, n_tok = _perturbed_forward(
                model, tokenizer, text, device, block, k, args.tau, eps,
                "hard", seed, args.max_length)
            soft_loss, soft_logits, soft_stats, _ = _perturbed_forward(
                model, tokenizer, text, device, block, k, args.tau, eps,
                "soft_edge", seed, args.max_length)
            hard_kl, hard_top1 = _logit_change(base_logits, hard_logits)
            soft_kl, soft_top1 = _logit_change(base_logits, soft_logits)
            rows.append({
                "n_tokens": n_tok,
                "base_nll": base_loss,
                "hard_delta_nll": hard_loss - base_loss,
                "soft_delta_nll": soft_loss - base_loss,
                "topk_flip_frac": stats["topk_flip_frac"],
                "hard_block_jump_rel": stats["hard_block_jump_rel"],
                "soft_block_jump_rel": soft_stats["soft_block_jump_rel"],
                "hard_logit_kl": hard_kl,
                "soft_logit_kl": soft_kl,
                "hard_top1_change_frac": hard_top1,
                "soft_top1_change_frac": soft_top1,
            })

        def mean(key):
            return sum(r[key] for r in rows) / len(rows)

        entry = {
            "eps": eps,
            "n_texts": len(rows),
            "topk_flip_frac": mean("topk_flip_frac"),
            "hard_block_jump_rel": mean("hard_block_jump_rel"),
            "soft_block_jump_rel": mean("soft_block_jump_rel"),
            "hard_delta_nll": mean("hard_delta_nll"),
            "soft_delta_nll": mean("soft_delta_nll"),
            "hard_logit_kl": mean("hard_logit_kl"),
            "soft_logit_kl": mean("soft_logit_kl"),
            "hard_top1_change_frac": mean("hard_top1_change_frac"),
            "soft_top1_change_frac": mean("soft_top1_change_frac"),
        }
        out["per_eps"].append(entry)
        print(f"  eps={eps:g} flip={entry['topk_flip_frac']:.3f} "
              f"hardJump={entry['hard_block_jump_rel']:.4f} "
              f"softJump={entry['soft_block_jump_rel']:.4f} "
              f"hardKL={entry['hard_logit_kl']:.4f} softKL={entry['soft_logit_kl']:.4f} "
              f"hardTop1={entry['hard_top1_change_frac']:.3f} "
              f"softTop1={entry['soft_top1_change_frac']:.3f}")
    return out


def run_section6(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    set_norm_topk(model, getattr(args, "norm_topk", "auto"))
    model.eval()
    k = args.k or getattr(model.config, "num_experts_per_tok", 2)
    blocks = discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("No MoE blocks (.gate+.experts) found in this model.")

    result = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "torch": torch.__version__,
    }
    if args.robustness or args.section6:
        result["robustness"] = run_robustness_loaded(args, model, tok, blocks, k, device)
    if args.quality or args.section6:
        result["regating_quality"] = run_regating_quality_loaded(args, model, tok, blocks, device)

    path = args.out or f"section6_{args.model.split('/')[-1]}.json"
    with open(path, "w") as f:
        json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}.")


# --------------------------------------------------------------------------- #
# Geometry experiment (normal vs tangent, k/k+1 swap, distance-to-tear)        #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _forward_capture(model, tokenizer, text, device, block, max_length):
    """Native forward; capture the target block's flat input h (T, H)."""
    enc = _encode_text(tokenizer, text, device, max_length)
    cap = {}

    def hook(mod, inp, out):
        cap["h"] = inp[0].reshape(-1, inp[0].shape[-1]).detach()
    handle = block.register_forward_hook(hook)
    try:
        o = model(**enc, labels=enc["input_ids"])
    finally:
        handle.remove()
    return o.logits.detach(), cap["h"]


@torch.no_grad()
def _forward_replace(model, tokenizer, text, device, block, y_replace, max_length):
    """Forward replacing the block's output with a precomputed y_replace (T, H)."""
    enc = _encode_text(tokenizer, text, device, max_length)

    def hook(mod, inp, out):
        h = inp[0]
        return _replace_first_tensor(out, y_replace.reshape_as(h).to(dtype=h.dtype))
    handle = block.register_forward_hook(hook)
    try:
        o = model(**enc, labels=enc["input_ids"])
    finally:
        handle.remove()
    return o.logits.detach()


def _block_jump(y_a, y_b):
    return ((y_a - y_b).norm(dim=-1).median()
            / (y_b.norm(dim=-1).median() + 1e-6)).item()


@torch.no_grad()
def run_geometry(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    set_norm_topk(model, getattr(args, "norm_topk", "auto"))
    model.eval()
    k = args.k or getattr(model.config, "num_experts_per_tok", 2)
    blocks = discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("No MoE blocks (.gate+.experts) found in this model.")
    idxs = ([int(x) for x in args.geom_layers.split(",")] if args.geom_layers
            else [min(max(args.layer, 0), len(blocks) - 1)])
    alphas = _parse_float_list(args.alphas)
    texts = SWEEP_PROBE_TEXTS[: (args.n_texts or 8)]
    result = {"model": args.model, "device": device, "dtype": str(dtype),
              "torch": torch.__version__, "k": k, "alphas": alphas,
              "n_texts": len(texts), "max_length": args.max_length, "layers": []}

    for idx in idxs:
        name, block, n_exp = blocks[idx]
        print(f"\n[geometry] layer={idx} {name} n_experts={n_exp} k={k}")
        dist_rel, dist_raw = [], []
        nt_rows = {a: {"normal": [], "tangent": []} for a in alphas}
        swap_rows = []
        for j, text in enumerate(texts):
            base_logits, h = _forward_capture(model, tok, text, device, block, args.max_length)
            geo = boundary_geometry(block, h, k)
            hn = h.norm(dim=-1).clamp_min(1e-6)
            dist_raw += geo["distance"].float().tolist()
            dist_rel += (geo["distance"] / hn).float().tolist()
            y_id = forward_hard(block, h, k)[0]
            id_logits = _forward_replace(model, tok, text, device, block, y_id, args.max_length)
            kl_id, top1_id = _logit_change(base_logits, id_logits)
            for a in alphas:
                for direction in ("normal", "tangent"):
                    gen = torch.Generator(device=device).manual_seed(
                        args.perturb_seed + 1000 * j + int(a * 100))
                    h_pert, fl = geom_perturb(block, h, k, a, direction, gen)
                    y_pert = forward_hard(block, h_pert, k)[0]
                    pert_logits = _forward_replace(model, tok, text, device, block,
                                                   y_pert, args.max_length)
                    kl_p, top1_p = _logit_change(base_logits, pert_logits)
                    nt_rows[a][direction].append({
                        "kk1_flip": fl["kk1_flip"].float().mean().item(),
                        "any_topk_flip": fl["any_topk_flip"].float().mean().item(),
                        "block_jump_rel": _block_jump(y_pert, y_id),
                        "logit_kl_bs": kl_p - kl_id,
                        "top1_change_bs": top1_p - top1_id,
                    })
            y_swap = forward_swap_kk1(block, h, k)
            swap_logits = _forward_replace(model, tok, text, device, block, y_swap, args.max_length)
            kl_s, top1_s = _logit_change(base_logits, swap_logits)
            swap_rows.append({
                "block_jump_rel": _block_jump(y_swap, y_id),
                "logit_kl_bs": kl_s - kl_id,
                "top1_change_bs": top1_s - top1_id,
            })

        def agg(rows, key):
            return sum(r[key] for r in rows) / len(rows)

        nt_summary = []
        for a in alphas:
            entry = {"alpha": a}
            for direction in ("normal", "tangent"):
                rows = nt_rows[a][direction]
                entry[direction] = {kk: agg(rows, kk) for kk in rows[0]}
            nt_summary.append(entry)
            nrm, tan = entry["normal"], entry["tangent"]
            print(f"  a={a:<4g} normal[flip={nrm['kk1_flip']:.3f} jump={nrm['block_jump_rel']:.4f} "
                  f"klBS={nrm['logit_kl_bs']:.4f}]  tangent[flip={tan['kk1_flip']:.3f} "
                  f"jump={tan['block_jump_rel']:.4f} klBS={tan['logit_kl_bs']:.4f}]")
        swap_summary = {kk: agg(swap_rows, kk) for kk in swap_rows[0]}
        print(f"  swap kk1: jump={swap_summary['block_jump_rel']:.4f} "
              f"klBS={swap_summary['logit_kl_bs']:.4f} top1BS={swap_summary['top1_change_bs']:.4f}")
        dist_t = torch.tensor(dist_rel)
        dist_stats = {"rel_median": dist_t.median().item(),
                      "rel_p10": dist_t.quantile(0.10).item(),
                      "rel_p90": dist_t.quantile(0.90).item(),
                      "raw_median": torch.tensor(dist_raw).median().item(),
                      "n_tokens": len(dist_rel)}
        print(f"  distance-to-tear (rel ||h||): median={dist_stats['rel_median']:.4f} "
              f"p10={dist_stats['rel_p10']:.4f} p90={dist_stats['rel_p90']:.4f}")
        result["layers"].append({"layer_idx": idx, "block_name": name, "n_experts": n_exp,
                                 "distance_to_tear": dist_stats,
                                 "normal_vs_tangent": nt_summary, "swap": swap_summary})

    # per-layer re-gating quality: re-gate ONLY each target layer, vs all-hard baseline
    texts_q = SWEEP_PROBE_TEXTS[: (args.n_texts or len(SWEEP_PROBE_TEXTS))]
    hardq = _lm_eval(model, tok, texts_q, device, args.max_length)
    plr = []
    for idx in idxs:
        handles = _install_soft_regating_hooks([blocks[idx]], args.tau)
        try:
            softq = _lm_eval(model, tok, texts_q, device, args.max_length)
        finally:
            _remove_hooks(handles)
        ratio = softq["ppl"] / max(hardq["ppl"], 1e-12)
        plr.append({"layer_idx": idx, "tau": args.tau, "hard_ppl": hardq["ppl"],
                    "soft_ppl": softq["ppl"], "ppl_ratio": ratio})
        print(f"  per-layer re-gate L{idx}: hard_ppl={hardq['ppl']:.2f} "
              f"soft_ppl={softq['ppl']:.2f} ratio={ratio:.3f}")
    result["per_layer_regate"] = plr

    path = args.out or f"geom_{args.model.split('/')[-1]}.json"
    with open(path, "w") as f:
        json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}.")


# --------------------------------------------------------------------------- #
# DeepSeek-decomposition: SwiGLU clamp sweep + outlier-conditioned tear        #
# --------------------------------------------------------------------------- #

def _install_hard_hooks(blocks, k):
    """Replace every MoE block's output with our hard top-k forward (which honours
    the global expert clamp). Baseline for the clamp's quality effect."""
    handles = []
    for _name, block, _n in blocks:
        def hook(mod, inp, out, k=k):
            h = inp[0]
            y, _ = forward_hard(mod, h.reshape(-1, h.shape[-1]), k)
            return _replace_first_tensor(out, y.reshape_as(h).to(dtype=h.dtype))
        handles.append(block.register_forward_hook(hook))
    return handles


def _intermediate_abs_percentiles(block, H, percentiles):
    """Data-driven clamp levels: |SwiGLU intermediate| (fused) or |expert out|
    (ModuleList) percentiles over a sample of experts/tokens."""
    exp = block.experts
    sample = H[: min(256, H.shape[0])]
    vals = []
    for e in range(min(_moe_num_experts(block), 4)):
        if isinstance(exp, nn.ModuleList):
            v = exp[e](sample)
        else:
            gate, up = F.linear(sample, exp.gate_up_proj[e]).chunk(2, dim=-1)
            v = exp.act_fn(gate) * up
        vals.append(v.reshape(-1).abs().float())
    allv = torch.cat(vals)
    return {p: allv.quantile(p).item() for p in percentiles}


def _pearson(a, b):
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return (a * b).sum().item() / (a.norm().item() * b.norm().item() + 1e-9)


@torch.no_grad()
def _outlier_tear_correlation(block, H, k):
    """Are high-norm (outlier) tokens closer to the tear boundary and do they jump
    more under a k/k+1 swap? Reports correlations only (hypothesis-agnostic)."""
    geo = boundary_geometry(block, H, k)
    norm_h = H.norm(dim=-1)
    y_id = forward_hard(block, H, k)[0]
    y_swap = forward_swap_kk1(block, H, k)
    jump = (y_swap - y_id).norm(dim=-1) / (y_id.norm(dim=-1) + 1e-6)
    return {
        "n_tokens": H.shape[0],
        "norm_vs_distance": _pearson(norm_h, geo["distance"]),
        "norm_vs_jump": _pearson(norm_h, jump),
        "distance_vs_jump": _pearson(geo["distance"], jump),
    }


@torch.no_grad()
def run_clamp_sweep(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading {args.model} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    set_norm_topk(model, getattr(args, "norm_topk", "auto"))
    model.eval()
    k = args.k or getattr(model.config, "num_experts_per_tok", 2)
    blocks = discover_moe_blocks(model)
    if not blocks:
        raise RuntimeError("No MoE blocks (.gate+.experts) found in this model.")
    idx = min(max(args.layer, 0), len(blocks) - 1)
    name, block, n_exp = blocks[idx]
    texts = SWEEP_PROBE_TEXTS[: (args.n_texts or len(SWEEP_PROBE_TEXTS))]
    print(f"\n[clamp-sweep] layer={idx} {name} n_experts={n_exp} k={k}")

    set_expert_clamp(None)
    H_bf = capture_hidden_states(model, tok, texts, block, device)   # model dtype
    pcts = [0.999, 0.99, 0.95, 0.9]

    # ---- PHASE 1: static geometry in fp32 (continuity's fine grid underflows in bf16) ----
    blk_dtype = next(block.parameters()).dtype
    block.float()
    H = H_bf.float()
    levels = _intermediate_abs_percentiles(block, H, pcts)
    clamp_list = [("none", None)] + [(f"p{int(p * 1000)}", levels[p]) for p in pcts]
    gen = torch.Generator().manual_seed(args.path_seed)   # CPU: find_boundary_pair's randint is CPU
    h_a, h_b, _meta = find_boundary_pair(block, H, k, mode="random",
                                         generator=gen, return_meta=True)
    static = {}
    for tag, c in clamp_list:
        set_expert_clamp(c)
        m2 = tear_magnitude(block, H, k)
        sig = continuity_signature(block, h_a, h_b, k=k, tau=args.tau)["hard_topk"]
        static[tag] = {
            "M2_tear_median": m2["tear_median_all"],          # normalized (scale-invariant)
            "cliff_abs_median": m2["cliff_abs_median"],        # absolute ||E_k - E_{k+1}||
            "expert_norm_abs_median": m2["expert_norm_abs_median"],
            "hardG": sig["growth"], "hardJump_rel": sig["jump_rel"],
            "hardJump_abs": sig["jump_abs"],                   # absolute ||Δy||
        }
    set_expert_clamp(None)
    block.to(blk_dtype)                                        # restore for native bf16 ppl forward

    # ---- PHASE 2: quality (ppl) at the model's native dtype, clamp via hard hooks ----
    native = _lm_eval(model, tok, texts, device, args.max_length)
    rows = []
    for tag, c in clamp_list:
        set_expert_clamp(c)
        handles = _install_hard_hooks(blocks, k)
        try:
            q = _lm_eval(model, tok, texts, device, args.max_length)
        finally:
            _remove_hooks(handles)
        set_expert_clamp(None)
        row = {"tag": tag, "clamp": c, **static[tag], "ppl": q["ppl"], "nll": q["nll"]}
        rows.append(row)
        cstr = "none" if c is None else f"{c:.3f}"
        print(f"  clamp={tag:<5} c={cstr:<7} hardG={row['hardG']:.1f}x M2={row['M2_tear_median']:.3f} "
              f"cliffABS={row['cliff_abs_median']:.3f} jumpABS={row['hardJump_abs']:.3f} ppl={q['ppl']:.2f}")
    base_nll = rows[0]["nll"]
    for r in rows:
        r["delta_nll"] = r["nll"] - base_nll

    set_expert_clamp(None)
    corr = _outlier_tear_correlation(block, H_bf, k)
    print(f"  outlier corr: ||h||~distance={corr['norm_vs_distance']:.3f} "
          f"||h||~swapJump={corr['norm_vs_jump']:.3f} "
          f"distance~swapJump={corr['distance_vs_jump']:.3f}")

    result = {"model": args.model, "device": device, "dtype": str(dtype),
              "torch": torch.__version__, "layer_idx": idx, "block_name": name,
              "n_experts": n_exp, "k": k, "tau": args.tau,
              "clamp_percentiles": {f"p{int(p * 1000)}": levels[p] for p in pcts},
              "native_ppl": native["ppl"], "hard_hook_base_nll": base_nll,
              "clamp_sweep": rows, "outlier_correlation": corr}
    path = args.out or f"clamp_{args.model.split('/')[-1]}.json"
    with open(path, "w") as f:
        json.dump(_json_safe(result), f, indent=2, ensure_ascii=False)
    print(f"\nWrote {path}.")


# --------------------------------------------------------------------------- #
# Self-test: build a mock MoE block (transformers-compatible interface) and run #
# every analysis path. Lets us validate the harness with NO model download.    #
# --------------------------------------------------------------------------- #

class _MockExpert(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))

    def forward(self, x):
        return self.net(x)


class _MockMoEBlock(nn.Module):
    """Mimics OlmoeSparseMoeBlock / MixtralSparseMoeBlock: .gate + .experts."""

    def __init__(self, hidden=32, n_experts=8):
        super().__init__()
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.ModuleList([_MockExpert(hidden) for _ in range(n_experts)])


def selftest():
    torch.manual_seed(0)
    print("SELF-TEST (mock MoE block, no download) — validates harness logic")
    print("=" * 70)
    K = 2
    block = _MockMoEBlock(hidden=32, n_experts=8)
    H = torch.randn(512, 32) * 1.5

    s1 = routing_stats(block, H, K)
    s2 = tear_magnitude(block, H, K)
    diag = soft_edge_diagnostics(block, H, tau=0.15)
    h_a, h_b, meta = find_boundary_pair(block, H, K, mode="targeted", return_meta=True)
    sig = continuity_signature(block, h_a, h_b, k=K)

    print(f"[M1] (boundary {K}/{K+1})", {k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in s1.items()})
    print("[M2]", s2)
    print("[soft-diag]", {k: round(v, 4) for k, v in diag.items()})
    print("[M3]", {k: round(v, 4) if isinstance(v, float) else v for k, v in meta.items()})
    print_continuity_signature(sig, indent="     ")

    # assertions: theory reproduces through the production code path
    g_hard = sig["hard_topk"]["growth"]
    g_soft = sig["soft_edge"]["growth"]
    g_ctl = sig["hard_tied(ctl)"]["growth"]
    assert g_hard > 4.0, f"expected hard routing to tear (growth>4), got {g_hard:.1f}x"
    assert g_soft < 2.0, f"expected soft_edge continuous (growth<2), got {g_soft:.1f}x"
    assert g_ctl < 2.0, (f"NEG CONTROL FAILED: tied experts tore (growth {g_ctl:.1f}x) "
                         f"=> the tear is a routing-arithmetic artefact, not real!")
    assert 0.0 <= s2["tear_median_all"] <= 1.0
    print("\nPASS:")
    print(f"  hard_topk      tears        (growth {g_hard:.1f}x)")
    print(f"  soft_edge      continuous   (growth {g_soft:.1f}x)   [continuity-guaranteed gate]")
    print(f"  hard_tied(ctl) neg-control  (growth {g_ctl:.1f}x)   <- tear NEEDS expert disagreement")
    print("Harness logic validated. Ready to point --model at a real MoE on GPU.")


def layer1_batch_selftest(n_seeds=10, n_paths=4, pool_size=768, k=2, tau=0.15):
    print("LAYER-1 BATCH SELF-TEST (mock MoE, no download)")
    print("=" * 70)
    print(f"seeds={n_seeds}, random_boundary_paths_per_seed={n_paths}, "
          f"targeted_paths_per_seed=1, pool_size={pool_size}, k={k}, tau={tau}")

    buckets = {
        "random boundary paths": {"hard_topk": [], "soft_edge": [], "hard_tied(ctl)": []},
        "targeted boundary paths": {"hard_topk": [], "soft_edge": [], "hard_tied(ctl)": []},
    }
    active, empty = [], []
    target_diffs = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        block = _MockMoEBlock(hidden=32, n_experts=8)
        H = torch.randn(pool_size, 32) * 1.5
        diag = soft_edge_diagnostics(block, H, tau=tau)
        active.append(diag["avg_active_experts"])
        empty.append(diag["empty_under_plain_relu_frac"])

        gen = torch.Generator().manual_seed(10_000 + seed)
        for _ in range(n_paths):
            h_a, h_b, _ = find_boundary_pair(block, H, k, mode="random",
                                             generator=gen, return_meta=True)
            sig = continuity_signature(block, h_a, h_b, k=k, tau=tau)
            for mode in buckets["random boundary paths"]:
                buckets["random boundary paths"][mode].append(sig[mode]["growth"])

        h_a, h_b, meta = find_boundary_pair(block, H, k, mode="targeted", return_meta=True)
        target_diffs.append(meta["set_diff"])
        sig = continuity_signature(block, h_a, h_b, k=k, tau=tau)
        for mode in buckets["targeted boundary paths"]:
            buckets["targeted boundary paths"][mode].append(sig[mode]["growth"])

    for label, growths in buckets.items():
        _print_growth_summary(label, growths)

    active_s = _summary(active)
    empty_s = _summary(empty)
    diff_s = _summary(target_diffs)
    print("\n[soft-edge diagnostics across pools]")
    print(f"  avg_active_experts            median={active_s['median']:.2f} "
          f"p95={active_s['p95']:.2f} min={active_s['min']:.2f} max={active_s['max']:.2f}")
    print(f"  empty_under_plain_relu_frac   median={empty_s['median']:.3f} "
          f"p95={empty_s['p95']:.3f} min={empty_s['min']:.3f} max={empty_s['max']:.3f}")
    print("\n[targeted-path diagnostics]")
    print(f"  top-k set_diff                median={diff_s['median']:.0f} "
          f"p95={diff_s['p95']:.0f} min={diff_s['min']:.0f} max={diff_s['max']:.0f}")

    random_g = buckets["random boundary paths"]
    target_g = buckets["targeted boundary paths"]
    checks = [
        ("random hard_topk median > 8x", _summary(random_g["hard_topk"])["median"] > 8.0),
        ("random soft_edge p95 < 2x", _summary(random_g["soft_edge"])["p95"] < 2.0),
        ("random hard_tied p95 < 2x", _summary(random_g["hard_tied(ctl)"])["p95"] < 2.0),
        ("targeted hard_topk median > 8x", _summary(target_g["hard_topk"])["median"] > 8.0),
        ("targeted soft_edge p95 < 2x", _summary(target_g["soft_edge"])["p95"] < 2.0),
        ("targeted hard_tied p95 < 2x", _summary(target_g["hard_tied(ctl)"])["p95"] < 2.0),
    ]
    for label, ok in checks:
        print(f"  check: {label:<34} {'PASS' if ok else 'FAIL'}")
        assert ok, label
    print("\nPASS: Layer-1 geometry is stable under random boundary paths and targeted pressure paths.")


def _loglog_slope(xs, ys):
    """OLS slope + R^2 of log10(ys) vs log10(xs). Slope is the scaling exponent p
    in max_quotient ~ T^p: p=1 => order-0 jump, p=0 => Lipschitz/continuous."""
    lx = [math.log10(x) for x in xs]
    ly = [math.log10(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    den = sum((x - mx) ** 2 for x in lx)
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / den
    intc = my - slope * mx
    ss_res = sum((y - (intc + slope * x)) ** 2 for x, y in zip(lx, ly))
    ss_tot = sum((y - my) ** 2 for y in ly)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, r2


def scaling_synth_selftest(n_seeds=12, n_paths=4, pool_size=768, k=2, tau=0.15,
                           resolutions=(250, 500, 1000, 2000, 4000, 8000, 16000),
                           out="scaling_synth.json"):
    """Scaling validation for the continuity signature, on a synthetic block whose
    discontinuity is exact and known (no download). The reframe claim is that
    growth = max_quotient[T_max]/max_quotient[T_min] = T_max/T_min for a genuine
    C0 jump *by construction* (independent of k/E/layer). We verify the underlying
    power law max_quotient ~ T^p directly: p~=1 (hard) vs p~=0 (soft/tied)."""
    print("CONTINUITY-SIGNATURE SCALING SELF-TEST (mock MoE, no download)")
    print("=" * 70)
    print(f"seeds={n_seeds}, random_paths_per_seed={n_paths}, pool_size={pool_size}, "
          f"k={k}, resolutions={resolutions}")
    modes = ("hard_topk", "soft_edge", "hard_tied(ctl)")
    acc = {m: [[] for _ in resolutions] for m in modes}
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        block = _MockMoEBlock(hidden=32, n_experts=8)
        H = torch.randn(pool_size, 32) * 1.5
        gen = torch.Generator().manual_seed(10_000 + seed)
        for _ in range(n_paths):
            h_a, h_b, _ = find_boundary_pair(block, H, k, mode="random",
                                             generator=gen, return_meta=True)
            sig = continuity_signature(block, h_a, h_b, resolutions=resolutions,
                                       k=k, tau=tau)
            for m in modes:
                for i, v in enumerate(sig[m]["max_quotient"]):
                    acc[m][i].append(v)
    per_mode = {}
    for m in modes:
        med = [_summary(c)["median"] for c in acc[m]]
        slope, r2 = _loglog_slope(resolutions, med)
        per_mode[m] = {"max_quotient_median": med, "slope": slope, "r2": r2}
        print(f"  {m:<16} slope(exponent)={slope:+.4f}  R^2={r2:.5f}  "
              f"maxq[{resolutions[0]}]={med[0]:.3f}  maxq[{resolutions[-1]}]={med[-1]:.1f}")
    payload = {
        "block": "synthetic _MockMoEBlock(hidden=32, n_experts=8)",
        "n_seeds": n_seeds, "n_paths": n_paths, "k": k, "tau": tau,
        "resolutions": list(resolutions),
        "refinement_ratio": resolutions[-1] / resolutions[0],
        "per_mode": per_mode,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out}")
    assert per_mode["hard_topk"]["slope"] > 0.9 and per_mode["hard_topk"]["r2"] > 0.99, \
        "hard routing should scale at exponent ~1 (order-0 jump)"
    assert per_mode["soft_edge"]["slope"] < 0.1 and per_mode["hard_tied(ctl)"]["slope"] < 0.1, \
        "continuity controls should be flat (exponent ~0)"
    print("PASS: hard routing scales at exponent ~1 (order-0 jump); controls flat (~0).")


def geom_selftest():
    """Known-answer test for the raw-logit boundary geometry (no download).

    Constructs an explicit gate so ek/ek1, the boundary normal W_k - W_{k+1},
    and the distance-to-tear are analytically known, then checks that a normal
    perturbation crosses the k/k+1 boundary while a tangent one of the SAME
    magnitude does not, and that the counterfactual swap is causal."""
    import math
    print("GEOMETRY SELF-TEST (mock block, known-answer) — raw-logit boundary")
    print("=" * 70)
    # H=4, E=3, k=1 -> boundary 1/2. Explicit gate rows are the standard basis.
    block = _MockMoEBlock(hidden=4, n_experts=3)
    with torch.no_grad():
        block.gate.weight.copy_(torch.tensor([[1., 0., 0., 0.],
                                              [0., 1., 0., 0.],
                                              [0., 0., 1., 0.]]))
    k = 1
    h = torch.tensor([[2.0, 1.5, 0.0, 5.0]])    # logits=[2,1.5,0] -> ek=0,ek1=1,margin=0.5
    geo = boundary_geometry(block, h, k)
    ek, ek1 = geo["ek"][0].item(), geo["ek1"][0].item()
    margin, dist = geo["logit_margin"][0].item(), geo["distance"][0].item()
    exp_dist = 0.5 / math.sqrt(2)               # margin / ||W_0 - W_1||
    print(f"  ek={ek} ek1={ek1} logit_margin={margin:.4f} "
          f"distance={dist:.4f} (expect {exp_dist:.4f})")
    assert ek == 0 and ek1 == 1, f"expected ek=0,ek1=1, got {ek},{ek1}"
    assert abs(margin - 0.5) < 1e-4, f"logit margin {margin}"
    assert abs(dist - exp_dist) < 1e-4, f"distance {dist} != {exp_dist}"

    _, sN2 = geom_perturb(block, h, k, alpha=2.0, direction="normal")
    _, sN05 = geom_perturb(block, h, k, alpha=0.5, direction="normal")
    gen = torch.Generator().manual_seed(0)
    _, sT2 = geom_perturb(block, h, k, alpha=2.0, direction="tangent", gen=gen)
    print(f"  normal  a=2.0 kk1_flip={bool(sN2['kk1_flip'][0])} (expect True)")
    print(f"  normal  a=0.5 kk1_flip={bool(sN05['kk1_flip'][0])} (expect False)")
    print(f"  tangent a=2.0 kk1_flip={bool(sT2['kk1_flip'][0])} (expect False)")
    assert bool(sN2["kk1_flip"][0]) is True, "normal a=2 must cross k/k+1"
    assert bool(sN05["kk1_flip"][0]) is False, "normal a=0.5 must NOT cross"
    assert bool(sT2["kk1_flip"][0]) is False, "tangent (same |Δ|) must NOT cross k/k+1"

    y_id = forward_hard(block, h, k)[0]
    jump = (forward_swap_kk1(block, h, k) - y_id).norm().item() / (y_id.norm().item() + 1e-6)
    print(f"  swap kk1 block_jump={jump:.4f} (expect > 0)")
    assert jump > 1e-3, f"swap should change output, got {jump}"
    for e in range(1, 3):                        # tied control: swap must be a no-op
        block.experts[e].load_state_dict(block.experts[0].state_dict())
    y_id2 = forward_hard(block, h, k)[0]
    jump_tied = (forward_swap_kk1(block, h, k) - y_id2).norm().item() / (y_id2.norm().item() + 1e-6)
    print(f"  swap kk1 block_jump (tied ctl)={jump_tied:.6f} (expect ~0)")
    assert jump_tied < 1e-5, f"tied swap must be a no-op, got {jump_tied}"

    print("\nPASS: raw-logit normal crosses k/k+1; tangent (same |Δ|) does not; "
          "distance = margin/||W_k-W_{k+1}||; swap is causal (tied control = no-op).")


def clamp_selftest():
    """Known-answer test for the SwiGLU/expert clamp (no download).

    The clamp must (a) cap expert activation magnitude and (b) shrink the block
    output jump in the presence of an outlier expert, while leaving the top-k
    routing set completely unchanged (clamp acts AFTER routing, on the experts)."""
    print("CLAMP SELF-TEST (mock block, known-answer) — magnitude down, routing fixed")
    print("=" * 70)
    torch.manual_seed(0)
    block = _MockMoEBlock(hidden=16, n_experts=6)
    with torch.no_grad():
        for p in block.experts[0].parameters():
            p.mul_(8.0)                              # expert 0 = outlier
    k = 2
    H = torch.randn(128, 16) * 1.5

    set_expert_clamp(None)
    base_mask = _topk_set_masks(_gate_logits(block, H), k)
    y_un = forward_hard(block, H, k)[0]
    e0_un = _expert_forward(block, 0, H)
    set_expert_clamp(0.5)
    clamp_mask = _topk_set_masks(_gate_logits(block, H), k)
    y_cl = forward_hard(block, H, k)[0]
    e0_cl = _expert_forward(block, 0, H)
    set_expert_clamp(None)

    routing_same = torch.equal(base_mask, clamp_mask)
    maxnorm_un, maxnorm_cl = y_un.norm(dim=-1).max().item(), y_cl.norm(dim=-1).max().item()
    e0_max = e0_cl.abs().max().item()
    print(f"  routing identical under clamp: {routing_same}")
    print(f"  expert-0 |out| max: unclamped={e0_un.abs().max():.3f} "
          f"clamped={e0_max:.3f} (cap 0.5)")
    print(f"  block ||y|| max:    unclamped={maxnorm_un:.3f} clamped={maxnorm_cl:.3f}")
    assert routing_same, "clamp must NOT change the top-k routing set"
    assert e0_max <= 0.5 + 1e-5, f"clamp bound violated: {e0_max}"
    assert maxnorm_cl < maxnorm_un, "clamp must reduce the block output magnitude"

    # redesign: clamp must reduce the ABSOLUTE expert cliff ||E_k - E_{k+1}||,
    # which is the DeepSeek mechanism (the normalized M2 is scale-invariant).
    set_expert_clamp(None)
    cliff_un = tear_magnitude(block, H, k)["cliff_abs_median"]
    set_expert_clamp(0.5)
    cliff_cl = tear_magnitude(block, H, k)["cliff_abs_median"]
    set_expert_clamp(None)
    print(f"  cliff_abs median: unclamped={cliff_un:.3f} clamped={cliff_cl:.3f}")
    assert cliff_cl < cliff_un, "clamp must reduce the ABSOLUTE expert cliff ||E_k - E_{k+1}||"
    print("\nPASS: clamp caps expert activation and shrinks the ABSOLUTE expert cliff / "
          "block jump while top-k routing is unchanged.")


DEFAULT_PROBE_TEXTS = [
    "The mitochondrion is the powerhouse of the cell.",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)",
    "In 1687 Newton published the Principia Mathematica.",
    "我们用张量方程把概念向量和逻辑关系组合起来。",
    "The integral of e^x is e^x plus a constant.",
    "Market volatility rose sharply after the earnings call.",
    "To be, or not to be, that is the question.",
    "Photosynthesis converts light energy into chemical energy.",
]


# Larger, domain-diverse corpus for the Layer-2 sweep so M1/M2 have enough
# tokens (fills the ~2048 boundary-pool) instead of the 8-sentence smoke set.
SWEEP_PROBE_TEXTS = DEFAULT_PROBE_TEXTS + [
    "class Heap:\n    def push(self, x):\n        self.data.append(x)\n        self._sift_up(len(self.data) - 1)",
    "The eigenvalues of a symmetric matrix are real and its eigenvectors orthogonal.",
    "Inflation eased to 2.4% in May as core services prices cooled across the eurozone.",
    "SELECT user_id, COUNT(*) FROM events WHERE ts > now() - interval '7 days' GROUP BY 1;",
    "李白的《将进酒》以奔放的笔触写尽人生须尽欢的豪情。",
    "The plaintiff bears the burden of proving each element by a preponderance of the evidence.",
    "Gradient descent updates parameters in the direction of steepest decrease of the loss.",
    "Mix two cups of flour with a teaspoon of baking soda before folding in the egg whites.",
    "El sistema inmunológico distingue lo propio de lo ajeno mediante receptores específicos.",
    "Quarterly free cash flow turned positive as capital expenditure normalised post-expansion.",
    "import torch\nx = torch.randn(8, 16)\ny = torch.nn.functional.softmax(x, dim=-1)",
    "The Treaty of Westphalia in 1648 established the principle of state sovereignty.",
    "Entropy of an ideal gas increases when it expands isothermally into a vacuum.",
    "「明日の天気は晴れ時々曇り、午後から雨が降るでしょう。」と予報士は述べた。",
    "A transformer attends over all positions, so its receptive field is global from layer one.",
    "Synaptic plasticity underlies learning by strengthening frequently coactivated connections.",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true",
                   help="validate harness logic on a mock block (no download)")
    p.add_argument("--layer1-batch", action="store_true",
                   help="multi-seed/multi-path Layer-1 geometry validation (no download)")
    p.add_argument("--geom-selftest", action="store_true",
                   help="known-answer test for raw-logit boundary geometry (no download)")
    p.add_argument("--scaling-synth", action="store_true",
                   help="continuity-signature scaling validation on a synthetic block (no download)")
    p.add_argument("--geom", action="store_true",
                   help="Deep-Manifold geometry: normal vs tangent, k/k+1 swap, distance-to-tear")
    p.add_argument("--alphas", default="0.25,0.5,1.0,2.0",
                   help="distance-scaled perturbation multipliers for --geom")
    p.add_argument("--geom-layers", default=None,
                   help="comma list of MoE block indices for --geom (default: --layer)")
    p.add_argument("--clamp-selftest", action="store_true",
                   help="known-answer test for the expert-activation clamp (no download)")
    p.add_argument("--clamp-sweep", action="store_true",
                   help="DeepSeek SwiGLU-clamp decomposition: M2/hardJump/hardG/ppl vs clamp")
    p.add_argument("--batch-seeds", type=int, default=10)
    p.add_argument("--batch-paths", type=int, default=4)
    p.add_argument("--batch-pool", type=int, default=768)
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--layer", type=int, default=8, help="which MoE block to probe")
    p.add_argument("--sweep", action="store_true",
                   help="Layer-2: load model once, sweep ALL MoE blocks, dump JSON")
    p.add_argument("--section6", action="store_true",
                   help="run robustness + re-gating quality experiments for the paper's section 6")
    p.add_argument("--robustness", action="store_true",
                   help="hidden-perturbation robustness: flips, jumps, logit/prediction changes")
    p.add_argument("--quality", action="store_true",
                   help="evaluate zero-retrain continuous re-gating quality")
    p.add_argument("--norm-topk", choices=["auto", "on", "off"], default="auto",
                   help="renorm gathered top-k weights: auto=honor model.config.norm_topk_prob "
                        "(OLMoE/Qwen ship false), on=force renorm, off=force native")
    p.add_argument("--sweep-layers", default=None,
                   help="comma list of block indices to sweep (default: all)")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--n-texts", type=int, default=None,
                   help="probe texts (default: 8 for single-layer, full corpus for --sweep)")
    p.add_argument("--max-length", type=int, default=128,
                   help="max token length for section-6 eval texts")
    p.add_argument("--tau", type=float, default=0.15, help="soft-edge threshold")
    p.add_argument("--k", type=int, default=None,
                   help="routing k (default: model.config.num_experts_per_tok)")
    p.add_argument("--path-seed", type=int, default=0,
                   help="seed for random boundary-path selection in real-model probing")
    p.add_argument("--m3-paths", type=int, default=8,
                   help="number of random boundary paths for M3a population diagnostic")
    p.add_argument("--perturb-eps", default="0,1e-4,1e-3,1e-2",
                   help="comma list of relative hidden perturbation magnitudes")
    p.add_argument("--perturb-seed", type=int, default=0,
                   help="seed for section-6 hidden perturbations")
    args = p.parse_args()

    if args.selftest:
        selftest()
    elif args.scaling_synth:
        scaling_synth_selftest()
    elif args.geom_selftest:
        geom_selftest()
    elif args.clamp_selftest:
        clamp_selftest()
    elif args.clamp_sweep:
        run_clamp_sweep(args)
    elif args.layer1_batch:
        layer1_batch_selftest(n_seeds=args.batch_seeds, n_paths=args.batch_paths,
                              pool_size=args.batch_pool, tau=args.tau)
    elif args.geom:
        run_geometry(args)
    elif args.section6 or args.robustness or args.quality:
        run_section6(args)
    elif args.sweep:
        run_sweep(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
