"""
Shared helpers for the golden-run dumpers. It runs INSIDE the reference image (mediaorquestator/diffrhythm2:1, upstream commit
7804f82), never on the host; see tools/README.md for the docker command.

What is patched, in memory only (the code in /opt/dr2 is not touched), and why:
  - CFM.sample_block_cache hard-codes `.half()` on the cache/text time tensors. With an fp32 model that makes the time embedding
    receive half inputs and fail, so the fp32 variant replaces the three `.half()` by `.float()`.
  - `torch.randn` for the per-block initial noise is replaced by pre-generated noise (NOISE[bid]): a C# port cannot reproduce
    PyTorch's random stream, so the noise is part of the fixture.
  - `odeint` is wrapped to record the time grid, the initial state and ALL the integration states of every block.
Each replacement is asserted by count, so an upstream change fails loudly instead of silently producing a different reference.

The fp64 variant (a calibration aid: how far fp32 is from the exact answer of the same network) uses `.double()`.
The fp16 variant keeps `.half()` (production behaviour) and only injects the noise.
"""
import hashlib
import inspect
import json
import os
import sys
import textwrap
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HOME", "/tmp")
UPSTREAM = "/opt/dr2"
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)

import torch  # noqa: E402

BLOCK = 10
CHANNELS = 64


def load_model(ckpt, precision="fp32"):
    """CFM(DiT) from config.json + model.safetensors, the way /opt/server.py builds it (sdpa attention, not flex)."""
    from safetensors.torch import load_file
    from diffrhythm2.backbones.dit import DiT
    from diffrhythm2.cfm import CFM

    with open(os.path.join(ckpt, "config.json")) as f:
        cfg = json.load(f)
    cfg["use_flex_attn"] = False
    model = CFM(transformer=DiT(**cfg), num_channels=cfg["mel_dim"], block_size=cfg["block_size"])
    model.load_state_dict(load_file(os.path.join(ckpt, "model.safetensors")))
    model = {"fp16": model.half, "fp64": model.double}.get(precision, model.float)()
    return model.eval(), cfg


def lyric_tokens(lyrics):
    """Flat token ids exactly as /opt/server.py builds them (parse_lyrics + sum(..., []))."""
    import inference as dr2

    dr2.lrc_tokenizer = dr2.CNENTokenizer()
    return sum(dr2.parse_lyrics(lyrics), [])


def style_embedding(ckpt, style_text):
    from muq import MuQMuLan

    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large", cache_dir=ckpt)
    with torch.no_grad():
        return mulan(texts=[style_text]).float()


def make_noise(seed, blocks):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randn(1, BLOCK, CHANNELS, generator=g) for _ in range(blocks)])  # [blocks, 1, 10, 64]


class Recorder:
    """Collects what odeint saw, one entry per block."""

    def __init__(self):
        self.start, self.grid, self.states = [], [], []

    def stacked(self):
        native = lambda x: x.float() if x.dtype == torch.float16 else x   # half is stored as float32; fp32 and fp64 keep their dtype
        return {
            "block_start": native(torch.stack(self.start)),   # [blocks, 1, 10, 64] initial state (the injected noise)
            "block_states": native(torch.stack(self.states)),  # [blocks, steps, 1, 10, 64] state at every grid point
            "time_grid": native(self.grid[0]),                 # [steps]
        }


def patched_sampler(model, noise, precision="fp32", scripted_blocks=None):
    """Returns (sample_fn, recorder). sample_fn has the signature of CFM.sample_block_cache.
    scripted_blocks: optional list of [1,10,64] tensors that replace the ODE result of each block (the rest of the sampler, caches and
    the end-of-song logic included, is the real code); used to exercise the EOS rule with controlled frames."""
    from diffrhythm2 import cfm
    from torchdiffeq import odeint as real_odeint

    src = textwrap.dedent(inspect.getsource(cfm.CFM.sample_block_cache))
    n_half = src.count(".half()")
    assert n_half == 2, f"upstream changed: expected 2 .half() in sample_block_cache, found {n_half}"
    if precision in ("fp32", "fp64"):
        src = src.replace(".half()", ".float()" if precision == "fp32" else ".double()")
    rand = "noisy_emb = torch.randn(batch, self.block_size, self.num_channels, device=device, dtype=style_prompt.dtype)"
    assert src.count(rand) == 1, "upstream changed: the noise line of sample_block_cache is not the expected one"
    src = src.replace(rand, "noisy_emb = NOISE[bid].to(device=device, dtype=style_prompt.dtype)")
    assert src.count("odeint(fn, noisy_emb, t_set, **self.odeint_kwargs)") == 1

    rec = Recorder()

    scripted = list(scripted_blocks) if scripted_blocks is not None else None

    def recording_odeint(fn, y0, t, **kw):
        if scripted is not None:
            out = torch.stack([y0, scripted.pop(0).to(y0.dtype)])  # outputs[-1] is the block the sampler keeps
        else:
            out = real_odeint(fn, y0, t, **kw)
        rec.start.append(y0.detach().clone())
        rec.grid.append(t.detach().clone())
        rec.states.append(out.detach().clone())
        return out

    ns = dict(cfm.__dict__)
    ns.update(NOISE=noise, odeint=recording_odeint)
    exec(compile(src, "<patched sample_block_cache>", "exec"), ns)
    return types.MethodType(ns["sample_block_cache"], model), rec


def end_pos(stream_blocks):
    """The end-of-song rule of sample_block_cache (cfm.py), applied block by block on the growing stream.
    stream_blocks: list of [1, 10, C] tensors in generation order. Returns (end_pos, blocks_sampled)."""
    stream = torch.zeros(1, 0, stream_blocks[0].shape[-1], dtype=stream_blocks[0].dtype)
    ep = 0
    for k, block in enumerate(stream_blocks):
        stream = torch.cat([stream, block], dim=1)
        pos = -1
        curr = stream[:, pos, :]
        eos = torch.ones_like(curr)
        last_kl = torch.nn.functional.mse_loss(curr, eos)
        if last_kl.abs() <= 0.05:
            while last_kl.abs() <= 0.05 and abs(pos) < stream.shape[1]:
                pos -= 1
                curr = stream[:, pos, :]
                last_kl = torch.nn.functional.mse_loss(curr, eos)
            return stream.shape[1] + pos, k + 1
        ep = stream.shape[1]
    return ep, len(stream_blocks)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir, title, notes, files):
    """MANIFEST.md in the format the core's FixtureDirectory verifies: a table whose first cell is `file` and which has a sha256 cell."""
    lines = [f"# {title}", "", notes.strip(), "", "| file | sha256 | bytes |", "|---|---|---|"]
    for name in sorted(files):
        p = os.path.join(out_dir, name)
        lines.append(f"| `{name}` | `{sha256(p)}` | {os.path.getsize(p)} |")
    with open(os.path.join(out_dir, "MANIFEST.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
