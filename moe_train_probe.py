"""
moe_train_probe.py

Layer-3 vehicle: a self-contained GPT-MoE trainer to test whether routing TEARS
show up as TRAINING instability and whether continuous routing changes it.
The static probe (moe_tear_probe.py) shows the tear EXISTS in weights; this
watches the tear DURING training and lets us intervene.

Hand-owned training loop so the instrumentation is exact. ReLU routing mirrors
ReMoE (Wang/Zhu/Chen, ICLR 2025, arXiv:2412.14711): relu(router_logits) used as
expert weights + adaptive L1 to steer the average active-expert count.

What the first smoke (notes 4.3) taught us, now built in for the scaled run:
  (A) real BPE corpus (char-level may be too smooth to spike) -- --data bpe
  (B) MATCHED avg_active across modes (relu via L1, soft via tau, both steer ->k)
  (C) deliberate spike provocation (--shift-at: vocab permute / corpus switch)
  (D) the causal intervention (--intervene-at / --intervene-on-spike: hard->soft live)
  (E) z-loss option + separated task/reg loss (separate tear vs numerical spikes)
  (F) tear-vs-training: at checkpoints, run the STATIC tear metrics on the live
      blocks -> does training heal or widen the seam?

Run (GPU). First do a short smoke, then the real run:
  python3 moe_train_probe.py --data bpe --routing hard --steps 300  --out smoke.json
  python3 moe_train_probe.py --data bpe --routing hard --steps 20000 --shift-at 8000 \
      --intervene-on-spike 0.3 --out run_hard.json
Compare hard vs relu vs soft with identical flags (same seed/data/shift).
"""

import argparse
import json
import math
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# static tear metrics reused on the LIVE training blocks (same repo).
# our MoEFeedForward exposes .gate (nn.Linear) + .experts (ModuleList), which the
# moe_tear_probe adapters already support, so the metrics apply unchanged.
from moe_tear_probe import (tear_magnitude, routing_stats, find_boundary_pair,
                            continuity_signature)


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
SHAKESPEARE = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "master/data/tinyshakespeare/input.txt")


def load_char(_args):
    if not os.path.exists("input.txt"):
        urllib.request.urlretrieve(SHAKESPEARE, "input.txt")
    text = open("input.txt").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


def _tokenize_hf(dataset, config, split, n_tokens, enc, cache):
    """Stream a HF text dataset, BPE-encode up to n_tokens, cache to uint16 .bin."""
    import numpy as np
    if os.path.exists(cache):
        return torch.from_numpy(np.fromfile(cache, dtype=np.uint16).astype("int64"))
    from datasets import load_dataset
    print(f"tokenizing {dataset}:{config}:{split} -> {n_tokens} tokens (first run, caches to {cache})")
    ds = load_dataset(dataset, config, split=split, streaming=True)
    buf, total = [], 0
    eot = enc.eot_token
    for ex in ds:
        t = ex.get("text") or ex.get("content") or ""
        if not t:
            continue
        buf.extend(enc.encode_ordinary(t)); buf.append(eot)
        total = len(buf)
        if total >= n_tokens:
            break
    arr = np.array(buf[:n_tokens], dtype=np.uint16)
    arr.tofile(cache)
    return torch.from_numpy(arr.astype("int64"))


def load_bpe(args):
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab = enc.n_vocab
    tag = args.hf_dataset.replace("/", "_")
    ids = _tokenize_hf(args.hf_dataset, args.hf_config, "train", args.n_tokens, enc,
                       f"corpus_{tag}.bin")
    n = int(0.95 * len(ids))
    train, val = ids[:n], ids[n:]
    # optional second corpus for a realistic distribution shift (mode=corpus)
    shift = None
    if args.shift_at >= 0 and args.shift_mode == "corpus":
        sds = args.shift_dataset or args.hf_dataset
        stag = sds.replace("/", "_")
        shift = _tokenize_hf(sds, args.shift_config, "train", args.n_tokens, enc,
                             f"corpus_{stag}_shift.bin")
    return train, val, vocab, shift


def get_batch(data, block, bs, device, gen, perm=None):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix]).to(device)
    if perm is not None:                                  # synthetic distribution shock
        x, y = perm[x], perm[y]
    return x, y


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head, block):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # flash-attention kernel when available
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MoEFeedForward(nn.Module):
    """MoE FFN, routing switchable at runtime (for the intervention). Returns
    (out, logits, w, l1) so the loop can do z-loss / L1 / active-count steering."""

    def __init__(self, d, hidden, n_exp, k, routing, tau):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
             for _ in range(n_exp)])
        self.gate = nn.Linear(d, n_exp, bias=False)
        self.shared = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.n_exp, self.k, self.routing, self.tau = n_exp, k, routing, tau
        self.tear_level = 1.0                             # (exp 2) 1=full tear, 0=experts tied to shared

    def _weights(self, logits):
        if self.routing == "hard":
            gates = logits.softmax(dim=-1)
            topv, topi = gates.topk(self.k, dim=-1)
            topv = topv / topv.sum(-1, keepdim=True)
            return torch.zeros_like(gates).scatter_(1, topi, topv), logits.new_zeros(())
        if self.routing == "relu":
            w = F.relu(logits)
            return w, w.mean()
        if self.routing == "soft":                        # continuity-guaranteed (matches static probe)
            gates = logits.softmax(dim=-1)
            gmax = gates.max(dim=-1, keepdim=True).values
            tau_eff = torch.minimum(torch.full_like(gmax, self.tau), 0.95 * gmax)
            w = F.relu(gates - tau_eff)                   # top expert weight >= 0.05*gmax > 0 => never empty
            return w / w.sum(-1, keepdim=True), logits.new_zeros(())
        raise ValueError(self.routing)

    def forward(self, x):
        B, T, C = x.shape
        h = x.reshape(-1, C)
        logits = self.gate(h)
        w, l1 = self._weights(logits)
        out = torch.zeros_like(h)
        for e in range(self.n_exp):
            m = w[:, e] > 0
            if m.any():
                out[m] += w[m, e:e + 1] * self.experts[e](h[m])
        if self.tear_level < 1.0:                         # (exp 2) blend toward shared expert -> dial the tear
            out = self.tear_level * out + (1.0 - self.tear_level) * w.sum(-1, keepdim=True) * self.shared(h)
        return out.view(B, T, C), logits, w, l1


class Block(nn.Module):
    def __init__(self, d, n_head, hidden, n_exp, k, routing, tau, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, block)
        self.ln2 = nn.LayerNorm(d)
        self.moe = MoEFeedForward(d, hidden, n_exp, k, routing, tau)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        m, logits, w, l1 = self.moe(self.ln2(x))
        return x + m, logits, w, l1


class GPTMoE(nn.Module):
    def __init__(self, vocab, d, n_head, n_layer, hidden, n_exp, k, routing, tau, block):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList(
            [Block(d, n_head, hidden, n_exp, k, routing, tau, block) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)
        self.head.weight = self.tok.weight                # weight tying (after init)

    @staticmethod
    def _init(m):                                         # nanoGPT-style init (avoid huge init loss)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_routing(self, routing):                       # (D) live intervention
        for blk in self.blocks:
            blk.moe.routing = routing

    def set_tear_level(self, level):                      # (exp 2) dial tear magnitude
        for blk in self.blocks:
            blk.moe.tear_level = level

    def set_tau(self, tau):
        for blk in self.blocks:
            blk.moe.tau = tau

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        l1, zloss, actives = x.new_zeros(()), x.new_zeros(()), []
        for blk in self.blocks:
            x, logits, w, b_l1 = blk(x)
            l1 = l1 + b_l1
            zloss = zloss + torch.logsumexp(logits, dim=-1).pow(2).mean()   # (E) router z-loss
            actives.append((w > 0).float().sum(-1).mean())
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, l1, zloss / len(self.blocks), torch.stack(actives).mean()


# --------------------------------------------------------------------------- #
# (F) tear-vs-training: run the STATIC metrics on a live block via a hook      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_tear(model, probe_x, k, tau, layer=-1):
    """Capture the chosen MoE block's input on probe_x, run M2 (cliff+cosine) and
    M3 (growth+jump) on it. Returns a compact dict; logs how the tear evolves."""
    model.eval()
    blk = list(model.blocks)[layer].moe
    grabbed = {}
    h = blk.register_forward_hook(lambda m, i, o: grabbed.__setitem__("h", i[0].detach()))
    model(probe_x)
    h.remove()
    H = grabbed["h"].reshape(-1, grabbed["h"].shape[-1]).float()
    blk = blk.float()
    m2 = tear_magnitude(blk, H, k)
    try:
        h_a, h_b, _ = find_boundary_pair(blk, H, k, mode="targeted", return_meta=True)
        sig = continuity_signature(blk, h_a, h_b, k=k, tau=tau)
        hardG = sig["hard_topk"]["growth"]; hardJump = sig["hard_topk"]["jump_rel"]
        softJump = sig["soft_edge"]["jump_rel"]
    except Exception:
        hardG = hardJump = softJump = float("nan")
    model.train()
    return {"tear_med": m2["tear_median_all"], "cos_kk1": m2["cos_kk1_median"],
            "hardG": round(hardG, 2), "hardJump": round(hardJump, 4),
            "softJump": round(softJump, 4)}


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train(args):
    device = args.device
    gen = torch.Generator().manual_seed(args.seed)
    torch.manual_seed(args.seed)
    loader = load_bpe if args.data == "bpe" else load_char
    loaded = loader(args)
    train_data, val_data, vocab = loaded[0], loaded[1], loaded[2]
    shift_data = loaded[3] if len(loaded) > 3 else None
    print(f"data={args.data} vocab={vocab} train_tokens={len(train_data)} device={device}")

    tau0 = args.soft_tau if args.soft_tau > 0 else 1.0 / args.n_exp
    model = GPTMoE(vocab, args.d, args.n_head, args.n_layer, args.hidden,
                   args.n_exp, args.k, args.routing, tau0, args.block).to(device)
    if args.tear_level < 1.0:
        model.set_tear_level(args.tear_level)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {n_params/1e6:.1f}M | routing={args.routing} n_exp={args.n_exp} "
          f"k={args.k} target_active={args.k} lr={args.lr} tear_level={args.tear_level}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.wd)
    use_amp = args.amp and device == "cuda"               # bf16 autocast (cuda only); probes stay fp32
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("bf16 autocast ON (TF32 matmul enabled)")

    probe_x, _ = get_batch(val_data, args.block, args.probe_bs, device, gen)
    perm = None
    if args.shift_at >= 0 and args.shift_mode == "permute":
        perm = torch.randperm(vocab, generator=gen).to(device)   # fixed vocab relabel

    lam, tau = args.l1_init, tau0
    log, prev_loss, prev_sets, intervened = [], None, None, False

    for step in range(args.steps):
        shifted = args.shift_at >= 0 and step >= args.shift_at
        src = shift_data if (shifted and shift_data is not None) else train_data
        x, y = get_batch(src, args.block, args.bs, device, gen,
                         perm=perm if (shifted and perm is not None) else None)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss, l1, zloss, avg_active = model(x, y)      # training step in bf16; probes stay fp32
            total = loss + (lam * l1 if args.routing == "relu" else 0.0) + args.zloss * zloss
        lv = loss.item()
        spike = max(0.0, lv - prev_loss) if (prev_loss is not None and math.isfinite(lv)) else 0.0
        if not math.isfinite(lv) or lv > args.stop_loss:   # diverged -> stop before backward, don't burn GPU
            print(f"step {step}: loss={lv} non-finite or > stop_loss={args.stop_loss} -> EARLY STOP (diverged)")
            log.append({
                "step": step, "loss": round(lv, 4) if math.isfinite(lv) else None,
                "spike": round(spike, 4), "task_loss": round(lv, 4) if math.isfinite(lv) else None,
                "zloss": round(zloss.item(), 4), "grad_norm": None,
                "avg_active": round(avg_active.item(), 3), "lam": round(lam, 6),
                "tau": round(tau, 5), "shifted": shifted,
                "routing": model.blocks[0].moe.routing, "diverged": True,
            })
            break
        opt.zero_grad(); total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        prev_loss = lv

        # (B) steer avg_active -> k:  relu via lambda, soft via tau
        if args.routing == "relu":
            lam *= (1 + args.steer) if avg_active.item() > args.k else (1 - args.steer)
            lam = float(min(args.l1_max, max(args.l1_min, lam)))
        elif args.routing == "soft":
            tau += args.steer * tau0 * (1 if avg_active.item() > args.k else -1)
            tau = float(min(0.9, max(1e-4, tau))); model.set_tau(tau)

        # (D) intervention: flip hard->soft at a step or when a spike fires
        if (not intervened and args.intervene_to and
                ((args.intervene_at >= 0 and step >= args.intervene_at) or
                 (args.intervene_on_spike > 0 and spike > args.intervene_on_spike))):
            model.set_routing(args.intervene_to); intervened = True
            print(f">>> step {step}: INTERVENTION routing -> {args.intervene_to} (spike={spike:.3f})")

        rec = {"step": step, "loss": round(lv, 4), "spike": round(spike, 4),
               "task_loss": round(lv, 4), "zloss": round(zloss.item(), 4),
               "grad_norm": round(gnorm.item(), 3), "avg_active": round(avg_active.item(), 3),
               "lam": round(lam, 6), "tau": round(tau, 5), "shifted": shifted,
               "routing": model.blocks[0].moe.routing}

        if step % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                model.eval()
                w_last = _last_block_input(model, probe_x)[2]   # [N,E] weights, last MoE block
                model.train()
            cur = w_last > 0                               # active-set mask on the fixed probe batch
            if prev_sets is not None and prev_sets.shape == cur.shape:
                rec["churn"] = round((cur != prev_sets).any(-1).float().mean().item(), 4)
            prev_sets = cur
            if args.tear_every and (step % args.tear_every == 0 or step == args.steps - 1):
                rec["tear"] = probe_tear(model, probe_x, args.k, tau)
            print(f"step {step:6d} | loss {lv:.4f} | spike {spike:.4f} | active {avg_active.item():.2f}"
                  f" | churn {rec.get('churn', float('nan')):.4f}"
                  f"{' | '+str(rec['tear']) if 'tear' in rec else ''}")
        log.append(rec)

        if args.ckpt_every and step > 0 and step % args.ckpt_every == 0:
            torch.save(model.state_dict(), f"{args.out}.ckpt{step}.pt")

    grad_probe = None
    if args.grad_probe:
        gx, gy = get_batch(train_data, args.block, args.bs, device, gen)
        grad_probe, gn = grad_path_smoothness(model, gx, gy, args.grad_alpha)
        print("\n[EXP1] loss smoothness along the gradient direction (param-space tear probe):")
        for m, r in grad_probe.items():
            print(f"  {m:<5} max_q={r['max_q']} growth={r['growth']}x  "
                  f"tokens_flipped_over_path={r['tokens_flipped_over_path']}")
        print("  hard growth >> 1 => the optimizer's step direction CROSSES routing tears")
        print("                      (non-smooth loss landscape => gradient prediction unreliable)")
        print("  soft growth ~1   => same checkpoint, continuous routing => smooth step direction")

    valid_losses = [r["loss"] for r in log if r.get("loss") is not None]
    spikes = torch.tensor([r.get("spike", 0.0) for r in log]) if log else torch.tensor([0.0])
    churns = [r["churn"] for r in log if "churn" in r]
    summary = {"routing": args.routing, "data": args.data, "steps": args.steps,
               "lr": args.lr, "shift_at": args.shift_at, "intervene_to": args.intervene_to,
               "final_loss": (round(float(torch.tensor(valid_losses)[-50:].mean()), 4)
                              if valid_losses else None),
               "max_spike": round(spikes.max().item(), 4),
               "n_spikes_gt_0.3": int((spikes > 0.3).sum().item()),
               "churn_median": round(float(torch.tensor(churns).median()), 4) if churns else None,
               "diverged": any(r.get("diverged") for r in log),
               "stopped_at_step": (log[-1]["step"] if log and log[-1].get("diverged") else args.steps),
               "n_params_M": round(n_params / 1e6, 2)}
    json.dump({"summary": summary, "args": vars(args), "grad_probe": grad_probe, "log": log},
              open(args.out, "w"))
    print("\nSUMMARY:", json.dumps(summary, indent=2), f"\nwrote {args.out}")


def grad_path_smoothness(model, x, y, alpha_max, grids=(200, 800, 3200)):
    """(Experiment 1) Parameter-space analog of the input-space tear probe.
    Take the optimizer's own descent direction d = -g/||g||; sweep loss(theta + a*d)
    on a FIXED batch over a refined alpha-grid. If the OPTIMIZATION PATH crosses a
    routing tear, loss(a) has a C0 jump -> the difference quotient |dL|/|da| DIVERGES
    as the grid refines (growth ~ refine factor). A smooth path keeps it bounded.
    Compares the SAME checkpoint under hard vs soft routing. This tests the actual
    mechanism by which a tear could destabilise training: a non-smooth loss landscape
    along the step direction makes the gradient's prediction unreliable."""
    orig = model.blocks[0].moe.routing
    model.zero_grad(set_to_none=True)
    _, loss, _, _, _ = model(x, y)
    loss.backward()
    params = [p for p in model.parameters() if p.grad is not None]
    theta0 = [p.detach().clone() for p in params]
    g = [p.grad.detach().clone() for p in params]
    gnorm = math.sqrt(sum(float((gi * gi).sum()) for gi in g)) + 1e-12
    d = [(-gi / gnorm) for gi in g]
    res = {}
    def set_theta(a):
        for p, t, di in zip(params, theta0, d):
            p.copy_(t + a * di)

    for mode in ("hard", "soft"):
        model.set_routing(mode)
        maxq = []
        with torch.no_grad():
            set_theta(0.0); first_set = _last_block_input(model, x)[2] > 0
            set_theta(alpha_max); last_set = _last_block_input(model, x)[2] > 0
            flips = int((first_set != last_set).any(-1).sum().item())
            for T in grids:
                alphas = torch.linspace(0, alpha_max, T, device=x.device)
                losses = []
                for a in alphas:
                    set_theta(float(a))
                    _, L, _, _, _ = model(x, y)
                    losses.append(float(L))
                Lt = torch.tensor(losses)
                maxq.append(float((Lt[1:] - Lt[:-1]).abs().max() / (alpha_max / (T - 1))))
        res[mode] = {"max_q": [round(m, 1) for m in maxq],
                     "growth": round(maxq[-1] / maxq[0], 1) if maxq[0] > 0 else float("nan"),
                     "tokens_flipped_over_path": flips}
    with torch.no_grad():
        for p, t in zip(params, theta0):
            p.copy_(t)
    model.set_routing(orig)
    model.zero_grad(set_to_none=True)
    return res, gnorm


def _last_block_input(model, idx):
    """Recompute up to (but not through) the last MoE: returns that block's MoE output
    components by running the full forward and re-deriving on its captured input."""
    grab = {}
    blk = model.blocks[-1].moe
    h = blk.register_forward_hook(lambda m, i, o: grab.__setitem__("x", i[0].detach()))
    model(idx)
    h.remove()
    return blk(grab["x"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routing", choices=["hard", "relu", "soft"], default="hard")
    p.add_argument("--data", choices=["char", "bpe"], default="char")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--n-exp", type=int, default=8)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stop-loss", type=float, default=float("inf"),
                   help="early-stop if loss is non-finite or exceeds this (edge-of-stability runs)")
    # corpus (bpe)
    p.add_argument("--hf-dataset", default="Salesforce/wikitext")
    p.add_argument("--hf-config", default="wikitext-103-raw-v1")
    p.add_argument("--n-tokens", type=int, default=20_000_000)
    # provocation / intervention
    p.add_argument("--shift-at", type=int, default=-1, help="step to inject a distribution shift (-1 off)")
    p.add_argument("--shift-mode", choices=["permute", "corpus"], default="permute")
    p.add_argument("--shift-dataset", default=None)
    p.add_argument("--shift-config", default=None)
    p.add_argument("--intervene-at", type=int, default=-1)
    p.add_argument("--intervene-on-spike", type=float, default=0.0)
    p.add_argument("--intervene-to", choices=["soft", "relu", "hard", ""], default="")
    # routing knobs
    p.add_argument("--soft-tau", type=float, default=0.0, help="0 => 1/n_exp")
    p.add_argument("--zloss", type=float, default=0.0, help="router z-loss coeff (0=off)")
    p.add_argument("--amp", action="store_true", default=True, help="bf16 autocast on cuda (default on)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="disable bf16 autocast")
    p.add_argument("--l1-init", type=float, default=1e-3)
    p.add_argument("--steer", type=float, default=0.02, help="active-count steering rate")
    p.add_argument("--l1-min", type=float, default=1e-6)
    p.add_argument("--l1-max", type=float, default=1.0)
    # logging
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--tear-level", type=float, default=1.0,
                   help="(exp 2) tear magnitude dial: 1=full MoE, 0=experts tied to a shared expert (no tear)")
    p.add_argument("--grad-probe", action="store_true",
                   help="(exp 1) after training, probe loss smoothness along the gradient direction (hard vs soft)")
    p.add_argument("--grad-alpha", type=float, default=1.0, help="param-space sweep distance for --grad-probe")
    p.add_argument("--tear-every", type=int, default=0, help="run static tear metrics every N steps (0=off)")
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--probe-bs", type=int, default=16)
    p.add_argument("--out", default="run.json")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
