"""
Shared helpers for the BigVGAN dumpers (change add-bigvgan-decoder). Runs INSIDE the reference image (see tools/README.md).
Nothing in /opt/dr2 is modified: the stage-by-stage forward below is a line-for-line copy of BigVGAN.forward that also records the signal
after every stage, and each dumper asserts that its final result is bit-identical to calling the model.
"""
import os
import sys

os.environ.setdefault("HOME", "/tmp")
UPSTREAM = "/opt/dr2"
if UPSTREAM not in sys.path:
    sys.path.insert(0, UPSTREAM)

import torch  # noqa: E402

CHUNK = 20
SAMPLES_PER_FRAME = 9600
KEEP_CHANNELS = 2   # stage signals are stored for the first channels only (the full tensors are tens of MB); per-channel RMS covers all of them
# The raw (channel-subset) signal is kept for these stages; the largest ones (up3..up5, act_post) only keep their per-channel RMS to keep the fixtures small.
RAW_STAGES = {"conv_pre", "up0", "up1", "up2", "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "conv_post"}


def load_generator(ckpt, precision="fp32"):
    from bigvgan.model import Generator

    gen = Generator(os.path.join(ckpt, "decoder.json"), os.path.join(ckpt, "decoder.bin"))
    gen = gen.half() if precision == "fp16" else gen.float()
    return gen.eval()


def staged_forward(bv, x):
    """BigVGAN.forward copied from /opt/dr2/bigvgan/model.py, returning (waveform, {name: signal})."""
    stages = {}
    x = bv.conv_pre(x)
    stages["conv_pre"] = x
    for i in range(bv.num_upsamples):
        for i_up in range(len(bv.ups[i])):
            x = bv.ups[i][i_up](x)
        stages[f"up{i}"] = x                                  # after the transposed convolution, before the residual blocks
        xs = None
        for j in range(bv.num_kernels):
            if xs is None:
                xs = bv.resblocks[i * bv.num_kernels + j](x)
            else:
                xs += bv.resblocks[i * bv.num_kernels + j](x)
        x = xs / bv.num_kernels
        stages[f"stage{i}"] = x                               # the signal the next stage receives
    x = bv.activation_post(x)
    stages["act_post"] = x
    x = bv.conv_post(x)
    stages["conv_post"] = x
    assert not bv.use_tanh_at_final
    x = torch.clamp(x, min=-1.0, max=1.0)
    return x, stages


def stage_entries(prefix, stages):
    """Fixture tensors for a chunk: the first channels of every stage and the RMS of every channel (float32)."""
    out = {}
    for name, t in stages.items():
        t = t.detach().float()
        if name in RAW_STAGES:
            out[f"{prefix}_{name}"] = (t if name == "conv_pre" else t[:, :KEEP_CHANNELS]).contiguous()
        out[f"{prefix}_{name}_rms"] = t.pow(2).mean(dim=2).sqrt()[0].contiguous()   # [C]
    return out
