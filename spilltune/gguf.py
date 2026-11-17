"""Dependency-free GGUF reader: just the facts spilltune needs to plan GPU/CPU placement.

Tensor sizes come from the gaps between tensor data offsets, so new quant types need no lookup table.
"""
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


@dataclass
class ModelInfo:
    path: str
    arch: str
    file_bytes: int
    n_layers: int
    expert_count: int = 0
    layer_bytes: list = field(default_factory=list)    # all tensors of blk.N
    expert_bytes: list = field(default_factory=list)   # the *_exps tensors of blk.N (MoE routed experts)
    other_bytes: int = 0                               # embeddings, output head, norms outside blocks
    meta: dict = field(default_factory=dict)           # small scalar metadata only

    @property
    def is_moe(self):
        return self.expert_count > 0 and sum(self.expert_bytes) > 0

    @property
    def name(self):
        return Path(self.path).name


class GGUFError(ValueError):
    pass


def read(path, max_array=64):
    """Read metadata + tensor table of a GGUF file. Large arrays (tokenizer vocab) are skipped."""
    path = str(path)
    with open(path, "rb") as f:
        def rd(fmt):
            n = struct.calcsize("<" + fmt)
            b = f.read(n)
            if len(b) != n:
                raise GGUFError("truncated GGUF header")
            return struct.unpack("<" + fmt, b)[0]

        def rs():
            return f.read(rd("Q")).decode("utf-8", "replace")

        def skip(t):
            if t == 8:
                f.seek(rd("Q"), 1)
            elif t == 9:
                et, n = rd("I"), rd("Q")
                if et in _SCALAR:
                    f.seek(n * struct.calcsize("<" + _SCALAR[et]), 1)
                else:
                    for _ in range(n):
                        skip(et)
            else:
                f.seek(struct.calcsize("<" + _SCALAR[t]), 1)

        def val(t):
            if t == 8:
                return rs()
            if t == 9:
                et, n = rd("I"), rd("Q")
                if n > max_array:
                    if et in _SCALAR:
                        f.seek(n * struct.calcsize("<" + _SCALAR[et]), 1)
                    else:
                        for _ in range(n):
                            skip(et)
                    return None
                return [val(et) for _ in range(n)]
            if t not in _SCALAR:
                raise GGUFError(f"unknown metadata type {t}")
            return rd(_SCALAR[t])

        if f.read(4) != b"GGUF":
            raise GGUFError(f"{path} is not a GGUF file")
        version = rd("I")
        if version < 2:
            raise GGUFError(f"GGUF v{version} is too old")
        n_tensors, n_kv = rd("Q"), rd("Q")
        meta = {}
        for _ in range(n_kv):
            k = rs()
            v = val(rd("I"))
            if v is not None:
                meta[k] = v
        tensors = []
        for _ in range(n_tensors):
            name = rs()
            nd = rd("I")
            for _ in range(nd):
                rd("Q")
            rd("I")  # ggml type
            tensors.append((name, rd("Q")))
        align = int(meta.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
    file_bytes = Path(path).stat().st_size

    arch = str(meta.get("general.architecture", "unknown"))
    order = sorted(tensors, key=lambda t: t[1])
    sizes = {}
    for i, (name, off) in enumerate(order):
        end = order[i + 1][1] if i + 1 < len(order) else file_bytes - data_start
        sizes[name] = end - off

    n_layers = int(meta.get(f"{arch}.block_count", 0))
    if not n_layers:
        idx = [int(m.group(1)) for n, _ in tensors for m in [re.match(r"blk\.(\d+)\.", n)] if m]
        n_layers = max(idx) + 1 if idx else 0
    layer = [0] * n_layers
    experts = [0] * n_layers
    other = 0
    for name, size in sizes.items():
        m = re.match(r"blk\.(\d+)\.(.*)", name)
        if m and int(m.group(1)) < n_layers:
            i = int(m.group(1))
            layer[i] += size
            if "_exps" in m.group(2):
                experts[i] += size
        else:
            other += size
    small = {k: v for k, v in meta.items() if not isinstance(v, list) and "chat_template" not in k}
    return ModelInfo(path=path, arch=arch, file_bytes=file_bytes, n_layers=n_layers,
                     expert_count=int(meta.get(f"{arch}.expert_count", 0) or 0),
                     layer_bytes=layer, expert_bytes=experts, other_bytes=other, meta=small)


# Speculative-decoding helper files: what llama-server's --spec-type should be for a given draft GGUF.
def draft_type(info):
    """Best guess of the --spec-type for a draft/helper GGUF, from its architecture and name."""
    s = (info.arch + " " + info.name).lower()
    if "dflash" in s:
        return "draft-dflash"
    if "eagle" in s:
        return "draft-eagle3"
    if "mtp" in s or "nextn" in s:
        return "draft-mtp"
    return "draft-simple"   # an ordinary small model of the same family


def helper_mismatch(model, draft):
    """A reason the helper does not belong to this model, or None. Helpers predict the main model's hidden
    states / tokens, so their width (and vocabulary) has to match."""
    def width(i, *keys):
        for k in keys:
            v = i.meta.get(f"{i.arch}.{k}")
            if v:
                return int(v)
        return None
    mw = width(model, "embedding_length")
    dw = width(draft, "embedding_length_out", "embedding_length")
    t = draft_type(draft)
    if t in ("draft-mtp", "draft-dflash", "draft-eagle3") and mw and dw and mw != dw:
        return f"helper width {dw} != model width {mw}: it was made for a different model"
    return None
