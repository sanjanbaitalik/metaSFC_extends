"""Data loading utilities for MetaSCFC."""
from .connectome_dataset import (
    ConnectomeDataset,
    SyntheticConnectomeDataset,
    load_fc_sc_arrays,
)
from .hcp_targets import (
    KNOWN_TARGETS,
    build_task_labels,
    load_hcp_behavior,
    load_task_target,
    resolve_target,
)

__all__ = [
    "ConnectomeDataset",
    "KNOWN_TARGETS",
    "SyntheticConnectomeDataset",
    "build_task_labels",
    "load_fc_sc_arrays",
    "load_hcp_behavior",
    "load_task_target",
    "resolve_target",
]
