"""Shared constants and paths for the synthetic router pipeline."""

from pathlib import Path


CLUSTER_NUM = 4
PATCH_NUM = 4
INPUT_DIM = 200
OUT_CHANNEL = 8
DEFAULT_EXPERT_NUM_OPTIONS = [2, 4, 8, 12]
DEFAULT_OPTIMIZERS = ["muon", "original"]
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SEARCH_SPACE_PATH = REPO_ROOT / "synthetic_router_search_space.yaml"
