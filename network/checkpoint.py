"""Checkpoint compatibility for the paper's module names."""

from collections import OrderedDict


LEGACY_PREFIXES = {
    **{f"dconv{i}": f"dc2m.dconv{i}" for i in range(1, 5)},
    "ms_fuse": "dc2m.ms_fuse",
    "ms_fuse_refined": "dc2m.ms_fuse_refined",
    "ccdm": "dc2m.cue_disentanglement",
    "cue_aux_head": "dc2m.cue_aux_head",
    **{f"tcm{i}": f"dc2m.acgc{i}" for i in range(1, 5)},
    "correction_flow_head": "cpfc",
}


def _rename_key(key):
    key = key.removeprefix("module.")
    prefix, dot, suffix = key.partition(".")
    return LEGACY_PREFIXES.get(prefix, prefix) + dot + suffix


def normalize_state_dict(checkpoint):
    """Unwrap common formats, strip DDP prefixes, and migrate old module names."""
    if isinstance(checkpoint, dict):
        for name in ("model_state_dict", "state_dict", "model"):
            if name in checkpoint and isinstance(checkpoint[name], dict):
                checkpoint = checkpoint[name]
                break

    result = OrderedDict()
    for key, value in checkpoint.items():
        new_key = _rename_key(key)
        if new_key in result:
            raise ValueError(f"Conflicting checkpoint keys map to {new_key!r}.")
        result[new_key] = value
    metadata = getattr(checkpoint, "_metadata", None)
    if metadata is not None:
        result._metadata = OrderedDict()
        for key, value in metadata.items():
            if key == "module":
                key = ""
            result._metadata[_rename_key(key)] = value
    return result
