"""ERST: Learning Interaction from Entity Replacement in Spatiotemporal Forecasting.

GitHub-ready full-model implementation.

This file intentionally contains ONLY the full ERST model. All development-only experimental branches have been removed. The full-model computation, module
construction order, initialization, optimizer, scheduler, normalization, masking,
and early-stopping logic are preserved so that runs with the same data, seed, and
hyperparameters follow the original full-model implementation.

Supported presets
-----------------
Grid datasets (PDFormer/LibCity protocol):
    NYCTaxi, CHIBike, T-Drive
Node datasets:
    PEMS03, PEMS04, PEMS07, PEMS08, KNOWAIR, SDWPF,
    MILAN_SMS, MILAN_CALL, MILAN_INTERNET,
    LARGEST_CA_2019, LARGEST_CA_2021,
    LARGEST_GBA_2019, LARGEST_GBA_2021,
    LARGEST_GLA_2019, LARGEST_GLA_2021,
    LARGEST_SD_2019, LARGEST_SD_2021

Examples
--------
python erst_github.py --dataset PEMS08 --data_path ./data/PEMS/PEMS08.npz
python erst_github.py --dataset NYCTaxi --data_path ./data/NYCTaxi/NYCTaxi.grid
python erst_github.py --run_all_grid --data_root ./data
python erst_github.py --run_all_node --data_root ./data
"""

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 0. Configuration
# ============================================================

class Config:
    DATA_ROOT = "./data"
    OUTPUT_ROOT = "./outputs"

    DATASET_NAME = "NYCTaxi"
    DATA_PATH = "./data/NYCTaxi/NYCTaxi.grid"
    SAVE_DIR = "./outputs/NYCTaxi"
    SAVE_PREFIX = "NYCTaxi_ERST_inflow"

    IN_FEAT = 1
    FLOW_INDEX = 0
    NUM_NODES = 75
    TARGET_NAME = "inflow"
    TARGET_INDEX = 0
    IS_GRID = True

    # Grid benchmark defaults are overwritten by set_dataset().
    IN_LEN = 6
    OUT_LEN = 1
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    D_MODEL = 64
    NUM_LAYERS = 2
    DROPOUT = 0.1

    NUM_TOD = 48
    INTERVAL_TEXT = "30min"
    METRIC_MASK_VAL = 10.0

    USE_LAST_RESIDUAL = True
    STABLE_INIT = True

    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    PATIENCE = 15
    CLIP_GRAD = 5.0
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()

GRID_DATASETS = ["NYCTaxi", "CHIBike", "T-Drive"]
NODE_DATASETS = [
    "PEMS03", "PEMS04", "PEMS07", "PEMS08",
    "KNOWAIR", "SDWPF",
    "MILAN_SMS", "MILAN_CALL", "MILAN_INTERNET",
    "LARGEST_CA_2019", "LARGEST_CA_2021",
    "LARGEST_GBA_2019", "LARGEST_GBA_2021",
    "LARGEST_GLA_2019", "LARGEST_GLA_2021",
    "LARGEST_SD_2019", "LARGEST_SD_2021",
]
ALL_DATASETS = GRID_DATASETS + NODE_DATASETS


def update_save_prefix():
    target_suffix = f"_{cfg.TARGET_NAME}" if cfg.IS_GRID else ""
    cfg.SAVE_PREFIX = (
        f"{cfg.DATASET_NAME}_ERST"
        f"_L{cfg.NUM_LAYERS}"
        f"_in{cfg.IN_LEN}_out{cfg.OUT_LEN}"
        f"{target_suffix}"
        f"_seed{cfg.SEED}"
    )


def _resolve_grid_target(dataset_name, target):
    target = str(target).strip().lower()
    if target == "auto":
        target = "outflow" if dataset_name == "CHIBike" else "inflow"
    if target not in ["inflow", "outflow"]:
        raise ValueError("target must be one of: auto, inflow, outflow")
    return target


def _path(*parts):
    return str(Path(cfg.DATA_ROOT).joinpath(*parts))


def _set_output_dir(dataset_name):
    cfg.SAVE_DIR = str(Path(cfg.OUTPUT_ROOT) / dataset_name)


def set_dataset(name, data_path="", target="auto"):
    """Configure one of the paper datasets without changing model hyperparameters."""
    aliases = {
        "nyc": "NYCTaxi",
        "nyct": "NYCTaxi",
        "nyctaxi": "NYCTaxi",
        "chi": "CHIBike",
        "chibike": "CHIBike",
        "td": "T-Drive",
        "tdrive": "T-Drive",
        "t-drive": "T-Drive",
        "t_drive": "T-Drive",
    }
    raw_name = str(name).strip()
    grid_name = aliases.get(raw_name.lower())

    if grid_name is not None:
        cfg.IS_GRID = True
        cfg.DATASET_NAME = grid_name
        cfg.IN_FEAT = 1
        cfg.IN_LEN = 6
        cfg.OUT_LEN = 1
        cfg.TRAIN_RATIO = 0.7
        cfg.VAL_RATIO = 0.1

        if grid_name == "NYCTaxi":
            cfg.DATA_PATH = _path("NYCTaxi", "NYCTaxi.grid")
            cfg.NUM_NODES = 75
            cfg.NUM_TOD = 48
            cfg.INTERVAL_TEXT = "30min"
            cfg.METRIC_MASK_VAL = 10.0
        elif grid_name == "CHIBike":
            cfg.DATA_PATH = _path("CHIBike", "CHIBike.grid")
            cfg.NUM_NODES = 270
            cfg.NUM_TOD = 48
            cfg.INTERVAL_TEXT = "30min"
            cfg.METRIC_MASK_VAL = 5.0
        elif grid_name == "T-Drive":
            cfg.DATA_PATH = _path("T-Drive", "T-Drive.grid")
            cfg.NUM_NODES = 1024
            cfg.NUM_TOD = 24
            cfg.INTERVAL_TEXT = "60min"
            cfg.METRIC_MASK_VAL = 10.0

        _set_output_dir(grid_name)
        if str(data_path).strip():
            cfg.DATA_PATH = str(data_path).strip()

        cfg.TARGET_NAME = _resolve_grid_target(grid_name, target)
        cfg.TARGET_INDEX = 0 if cfg.TARGET_NAME == "inflow" else 1
        cfg.FLOW_INDEX = cfg.TARGET_INDEX
        update_save_prefix()
        return

    name = raw_name.upper()
    cfg.DATASET_NAME = name
    cfg.IS_GRID = False
    cfg.TARGET_NAME = "flow"
    cfg.TARGET_INDEX = 0
    cfg.IN_LEN = 12
    cfg.OUT_LEN = 12
    cfg.TRAIN_RATIO = 0.6
    cfg.VAL_RATIO = 0.2
    cfg.METRIC_MASK_VAL = 0.0

    if name == "PEMS03":
        cfg.DATA_PATH = _path("PEMS", "PEMS03.npz")
        cfg.NUM_NODES = 358
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "PEMS04":
        cfg.DATA_PATH = _path("PEMS", "PEMS04.npz")
        cfg.NUM_NODES = 307
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "PEMS07":
        cfg.DATA_PATH = _path("PEMS", "PEMS07.npz")
        cfg.NUM_NODES = 883
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "PEMS08":
        cfg.DATA_PATH = _path("PEMS", "PEMS08.npz")
        cfg.NUM_NODES = 170
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "KNOWAIR":
        cfg.DATA_PATH = _path("KnowAir", "KnowAir_PM25.npy")
        cfg.NUM_NODES = 184
        cfg.NUM_TOD = 8
        cfg.INTERVAL_TEXT = "3h"
    elif name == "SDWPF":
        cfg.DATA_PATH = _path("SDWPF", "SDWPF_Patv.npy")
        cfg.NUM_NODES = 134
        cfg.NUM_TOD = 144
        cfg.INTERVAL_TEXT = "10min"
    elif name == "MILAN_SMS":
        cfg.DATA_PATH = _path("Milan", "milan_400_sms.npy")
        cfg.NUM_NODES = 400
        cfg.NUM_TOD = 24
        cfg.INTERVAL_TEXT = "1h"
    elif name == "MILAN_CALL":
        cfg.DATA_PATH = _path("Milan", "milan_400_call.npy")
        cfg.NUM_NODES = 400
        cfg.NUM_TOD = 24
        cfg.INTERVAL_TEXT = "1h"
    elif name == "MILAN_INTERNET":
        cfg.DATA_PATH = _path("Milan", "milan_400_internet.npy")
        cfg.NUM_NODES = 400
        cfg.NUM_TOD = 24
        cfg.INTERVAL_TEXT = "1h"
    elif name == "LARGEST_CA_2019":
        cfg.DATA_PATH = _path("LargeST", "ca", "ca_his_2019.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "LARGEST_CA_2021":
        cfg.DATA_PATH = _path("LargeST", "ca", "ca_his_2021.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "LARGEST_GBA_2019":
        cfg.DATA_PATH = _path("LargeST", "gba", "gba_his_2019.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "LARGEST_GBA_2021":
        cfg.DATA_PATH = _path("LargeST", "gba", "gba_his_2021.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "LARGEST_GLA_2019":
        cfg.DATA_PATH = _path("LargeST", "gla", "gla_his_2019.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "LARGEST_GLA_2021":
        cfg.DATA_PATH = _path("LargeST", "gla", "gla_his_2021.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "LARGEST_SD_2019":
        cfg.DATA_PATH = _path("LargeST", "sd", "sd_his_2019.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    elif name == "LARGEST_SD_2021":
        cfg.DATA_PATH = _path("LargeST", "sd", "sd_his_2021.h5")
        cfg.NUM_NODES = None
        cfg.NUM_TOD = 288
        cfg.INTERVAL_TEXT = "5min"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    cfg.IN_FEAT = 1
    cfg.FLOW_INDEX = 0
    _set_output_dir(name)
    if str(data_path).strip():
        cfg.DATA_PATH = str(data_path).strip()
    update_save_prefix()


# ============================================================
# 1. Utilities
# ============================================================

def hr(char="─", width=92):
    print(char * width)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_mem_stat(tag=""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024 ** 2
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    reserved = torch.cuda.memory_reserved() / 1024 ** 2
    print(f"  [MEM {tag}] alloc={alloc:.1f}MB  peak={peak:.1f}MB  reserved={reserved:.1f}MB")


# ============================================================
# 2. Dataset
# ============================================================

class SpatiotemporalDataset(Dataset):
    def __init__(self, data, mean, std, in_len, out_len, tod_offset=0):
        data = data[:, :, cfg.FLOW_INDEX:cfg.FLOW_INDEX + 1]
        self.x = (data - mean) / (std + 1e-8)
        self.in_len = in_len
        self.out_len = out_len
        self.tod_off = tod_offset

    def __len__(self):
        return len(self.x) - self.in_len - self.out_len + 1

    def __getitem__(self, idx):
        x = self.x[idx: idx + self.in_len]
        y = self.x[idx + self.in_len: idx + self.in_len + self.out_len, :, 0:1]
        tod = (self.tod_off + idx + self.in_len - 1) % cfg.NUM_TOD
        return torch.FloatTensor(x), torch.FloatTensor(y), torch.LongTensor([tod])


def _load_libcity_grid(path):
    """Load LibCity/PDFormer grid files into [T, N, 2] = [inflow, outflow]."""
    usecols = ["time", "row_id", "column_id", "inflow", "outflow"]
    df = pd.read_csv(path, usecols=usecols)

    cells = (
        df[["row_id", "column_id"]]
        .drop_duplicates()
        .sort_values(["row_id", "column_id"])
        .reset_index(drop=True)
    )
    cells["cell_id"] = np.arange(len(cells), dtype=np.int64)

    times = np.array(sorted(df["time"].unique()))
    T = len(times)
    N = len(cells)
    expected = T * N

    if N != cfg.NUM_NODES:
        print(f"[Warn] cfg.NUM_NODES={cfg.NUM_NODES}, detected N={N}; using detected N.")
        cfg.NUM_NODES = N

    print(f"[Load] Grid rows={len(df):,}  T={T:,}  N={N:,}  expected={expected:,}")

    df_sorted = df.sort_values(["time", "row_id", "column_id"]).reset_index(drop=True)
    if len(df_sorted) == expected:
        return (
            df_sorted[["inflow", "outflow"]]
            .to_numpy(dtype=np.float32)
            .reshape(T, N, 2)
        )

    time_to_id = {t: i for i, t in enumerate(times)}
    df2 = df.merge(cells, on=["row_id", "column_id"], how="left", sort=False)
    t_code = df2["time"].map(time_to_id).to_numpy(dtype=np.int64)
    n_code = df2["cell_id"].to_numpy(dtype=np.int64)

    raw = np.zeros((T, N, 2), dtype=np.float32)
    raw[t_code, n_code, 0] = df2["inflow"].to_numpy(dtype=np.float32)
    raw[t_code, n_code, 1] = df2["outflow"].to_numpy(dtype=np.float32)
    print("[Warn] Missing/irregular grid rows detected; missing values filled with 0.")
    return raw


def load_data(path, batch_size):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Use --data_path for a single run or --data_root for the repository data root."
        )

    ext = os.path.splitext(path)[1].lower()

    if ext in [".grid", ".csv", ".txt"]:
        if not cfg.IS_GRID:
            raise ValueError(".grid/.csv/.txt loading is available only for a grid preset.")
        raw = _load_libcity_grid(path)
    elif ext in [".h5", ".hdf5"]:
        df = pd.read_hdf(path)
        raw = df.to_numpy(dtype=np.float32)
        del df
    else:
        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "data" not in loaded.files:
                    raise KeyError(f"NPZ file has no 'data' key. Available keys: {loaded.files}")
                raw = loaded["data"].astype(np.float32)
            finally:
                loaded.close()
        else:
            raw = np.asarray(loaded, dtype=np.float32)

    if raw.ndim == 2:
        raw = raw[..., None]
    if raw.ndim != 3:
        raise ValueError(f"Expected [T,N,F], got {raw.shape}")

    T, N, F_raw = raw.shape

    if cfg.NUM_NODES is None:
        cfg.NUM_NODES = int(N)
        print(f"[Data] Auto-inferred NUM_NODES={cfg.NUM_NODES}")
    elif N != cfg.NUM_NODES:
        raise ValueError(f"NUM_NODES mismatch: cfg.NUM_NODES={cfg.NUM_NODES}, raw N={N}")

    if cfg.FLOW_INDEX < 0 or cfg.FLOW_INDEX >= F_raw:
        raise ValueError(f"FLOW_INDEX={cfg.FLOW_INDEX} is invalid for rawF={F_raw}")
    if cfg.IS_GRID and F_raw < 2:
        raise ValueError(f"Grid data must contain [inflow,outflow], but rawF={F_raw}.")

    t1 = int(T * cfg.TRAIN_RATIO)
    t2 = int(T * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))

    target = raw[:, :, cfg.FLOW_INDEX]
    mean = target[:t1].mean()
    std = target[:t1].std()

    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
        raise ValueError(f"Invalid normalization statistics: mean={mean}, std={std}")

    print(
        f"[Data] dataset={cfg.DATASET_NAME}  path={path}\n"
        f"[Data] T={T}  N={N}  rawF={F_raw}  train/val/test={t1}/{t2-t1}/{T-t2}"
    )
    if cfg.IS_GRID:
        print(
            f"[Data] Grid protocol: split=7:1:2, past {cfg.IN_LEN} -> next {cfg.OUT_LEN}, "
            f"target={cfg.TARGET_NAME}, interval={cfg.INTERVAL_TEXT}"
        )
        print(
            f"[Norm] TRAIN-only mean={mean:.4f} std={std:.4f}; "
            f"metric/loss mask true >= {cfg.METRIC_MASK_VAL:g}"
        )
    else:
        print(
            f"[Data] Node protocol: split={cfg.TRAIN_RATIO:.1f}:{cfg.VAL_RATIO:.1f}:"
            f"{1-cfg.TRAIN_RATIO-cfg.VAL_RATIO:.1f}, {cfg.IN_LEN}->{cfg.OUT_LEN}, "
            f"mean={mean:.4f}, std={std:.4f}, interval={cfg.INTERVAL_TEXT}"
        )

    def make(split, offset, shuffle):
        ds = SpatiotemporalDataset(split, mean, std, cfg.IN_LEN, cfg.OUT_LEN, offset)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
        )

    return (
        make(raw[:t1], 0, True),
        make(raw[t1:t2], t1, False),
        make(raw[t2:], t2, False),
        mean,
        std,
    )


# ============================================================
# 3. ERST model
# ============================================================

class FlowProjection(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.proj = nn.Linear(in_feat, 1, bias=False)
        if in_feat == 1:
            nn.init.ones_(self.proj.weight)
        else:
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        return self.proj(x).squeeze(-1)


class TemporalEncoder(nn.Module):
    def __init__(self, in_len, d_model, num_tod=288, dropout=0.1):
        super().__init__()
        self.ip = nn.Linear(in_len, d_model)
        self.te = nn.Embedding(num_tod, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_flow, tod, return_time_embedding=False):
        h = self.ip(x_flow.permute(0, 2, 1))
        te_base = self.te(tod.squeeze(-1))
        te = te_base.unsqueeze(1).expand(-1, x_flow.shape[2], -1)
        h = torch.cat([h, te], dim=-1)
        out = self.norm(self.mlp(h))
        if return_time_embedding:
            return out, te_base
        return out


class EntityFeedForwardLayer(nn.Module):
    """Entity-wise feedforward transformation g_ffn^(l) from the paper."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.ffn(x))


class EntityReplacementResponseLayer(nn.Module):
    r"""Full ERST interaction layer.

    For entity i:
        u_i = phi(h_i)
        mu_-i = mean_{j != i} u_j
        b_i = psi([s_i, mu_-i, tau_t, d_l])
        H_rep^i = H with h_i replaced by b_i
        r_i = F(H) - F(H_rep^i)

    The implementation computes all entity replacements in O(N) entity-wise work:
        mu_rep,i = mu + (phi(b_i) - phi(h_i)) / N.

    The reference does not use static entity identity. Entity identity is used only
    to modulate the response when the response is incorporated into the entity state.
    """

    def __init__(self, d_model, num_layers, dropout=0.1):
        super().__init__()
        D = int(d_model)
        L = max(1, int(num_layers))

        self.depth_emb = nn.Embedding(L, D)

        # psi([s_i, mu_-i, tau_t, d_l]) -> b_i
        self.reference_mlp = nn.Sequential(
            nn.LayerNorm(D * 4),
            nn.Linear(D * 4, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
        )

        # Entity/time/layer information modulates only the response.
        self.response_modulator = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.Sigmoid(),
        )

        # phi
        self.entity_map = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
        )

        # rho, i.e., F(H)=rho(mean(phi(h_j)))
        self.representation_function = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D * 2),
            nn.GELU(),
            nn.Linear(D * 2, D),
        )

        # g_r
        self.response_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
        )

        # g_u and alpha
        self.node_update = nn.Sequential(
            nn.Linear(D * 2, D * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D * 2, D),
        )
        self.response_gate = nn.Sequential(
            nn.Linear(D * 2, D),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(D)

    def forward(self, h, history_e, entity_e, tod_e, layer_idx=0):
        B, N, D = h.shape
        if N < 1:
            raise ValueError("ERST requires at least one entity.")
        if history_e.shape != h.shape:
            raise ValueError(
                f"history_e shape must match h: history_e={tuple(history_e.shape)}, h={tuple(h.shape)}"
            )

        # 1) Factual transformed entities and leave-one-out information.
        u = self.entity_map(h)
        sum_u = u.sum(dim=1, keepdim=True)
        mean_original = sum_u / float(N)
        if N > 1:
            other_entity_mean = (sum_u - u) / float(N - 1)
        else:
            other_entity_mean = torch.zeros_like(u)

        # 2) Reference b_i from history, other entities, time, and layer.
        time_ref = tod_e.unsqueeze(1).expand(-1, N, -1)
        depth_index = min(int(layer_idx), self.depth_emb.num_embeddings - 1)
        depth_id = torch.tensor(depth_index, device=h.device, dtype=torch.long)
        depth_ref = self.depth_emb(depth_id).view(1, 1, D).expand(B, N, D)

        reference_context = torch.cat(
            [history_e, other_entity_mean, time_ref, depth_ref], dim=-1
        )
        reference = self.reference_mlp(reference_context)

        # Static entity identity is deliberately excluded from reference construction.
        identity_seed = entity_e + time_ref + depth_ref
        response_modulation = self.response_modulator(identity_seed)

        # 3) Replace one entity representation while keeping entity count and
        #    normalization unchanged, then evaluate the same function rho.
        u_ref = self.entity_map(reference)
        mean_replaced = mean_original + (u_ref - u) / float(N)

        f_original = self.representation_function(mean_original)
        f_replaced = self.representation_function(mean_replaced)
        response = f_original - f_replaced

        # 4) Use the response for the corresponding entity update.
        response_feat = self.response_proj(response) * response_modulation
        node_input = torch.cat([h, response_feat], dim=-1)
        update = self.node_update(node_input)
        alpha = self.response_gate(node_input)
        return self.norm(h + alpha * update)


class ERST(nn.Module):
    """Entity Replacement for Spatiotemporal Forecasting (full model only)."""

    def __init__(self):
        super().__init__()
        D = cfg.D_MODEL

        # Keep this construction order identical to the original full model.
        self.flow_proj = FlowProjection(cfg.IN_FEAT)
        self.temporal_enc = TemporalEncoder(
            in_len=cfg.IN_LEN,
            d_model=D,
            num_tod=cfg.NUM_TOD,
            dropout=cfg.DROPOUT,
        )
        self.entity_emb = nn.Embedding(cfg.NUM_NODES, D)
        self.layers = nn.ModuleList([
            EntityFeedForwardLayer(D, cfg.DROPOUT)
            for _ in range(cfg.NUM_LAYERS)
        ])
        self.response_layer = EntityReplacementResponseLayer(
            d_model=D,
            num_layers=max(1, cfg.NUM_LAYERS),
            dropout=cfg.DROPOUT,
        )
        self.decoder = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(D, cfg.OUT_LEN),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.flow_proj.proj:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

        if cfg.STABLE_INIT:
            last_linear = self.decoder[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def _entity_identity(self, batch_size, n_entities, device):
        if n_entities != cfg.NUM_NODES:
            raise ValueError(
                f"Entity embedding requires N={cfg.NUM_NODES}, got N={n_entities}."
            )
        entity_id = torch.arange(n_entities, device=device)
        return self.entity_emb(entity_id).unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x, tod):
        # x: (B,T,N,F), output: (B,OUT,N,1)
        x_flow = self.flow_proj(x)
        h, tod_e = self.temporal_enc(x_flow, tod, return_time_embedding=True)
        history_e = h
        entity_e = self._entity_identity(x.shape[0], h.shape[1], x.device)

        for idx, layer in enumerate(self.layers):
            h = layer(h)
            h = self.response_layer(
                h,
                history_e=history_e,
                entity_e=entity_e,
                tod_e=tod_e,
                layer_idx=idx,
            )

        delta = self.decoder(h)
        if cfg.USE_LAST_RESIDUAL:
            last = x_flow[:, -1, :].unsqueeze(-1)
            pred = last + delta
        else:
            pred = delta

        return pred.permute(0, 2, 1).unsqueeze(-1)


# ============================================================
# 4. Metrics
# ============================================================

def metric_mask(true, null_val=0.0):
    if cfg.IS_GRID:
        true2 = true.clone()
        true2[torch.abs(true2) < 1e-4] = 0.0
        return ((true2 != 0.0) & (true2 >= float(cfg.METRIC_MASK_VAL))).float()
    return (true.abs() > null_val).float()


def masked_mae(pred, true, null_val=0.0):
    mask = metric_mask(true, null_val)
    return (torch.abs(pred - true) * mask).sum() / (mask.sum() + 1e-8)


def masked_rmse(pred, true, null_val=0.0):
    mask = metric_mask(true, null_val)
    return torch.sqrt(((pred - true) ** 2 * mask).sum() / (mask.sum() + 1e-8))


def masked_mape(pred, true, null_val=1.0):
    mask = metric_mask(true, null_val)
    return (
        ((pred - true).abs() / (true.abs() + 1e-8) * mask).sum()
        / (mask.sum() + 1e-8)
    ) * 100.0


# ============================================================
# 5. Train / evaluation
# ============================================================

def train_epoch(model, loader, optimizer, mean, std, device):
    model.train()
    total_mae = 0.0
    count = 0

    mt = torch.tensor(mean, device=device, dtype=torch.float32)
    st = torch.tensor(std, device=device, dtype=torch.float32)

    for x, y, tod in loader:
        x, y, tod = x.to(device), y.to(device), tod.to(device)
        optimizer.zero_grad(set_to_none=True)

        out = model(x, tod)
        pred_r = out * st + mt
        y_r = y * st + mt
        loss = masked_mae(pred_r, y_r)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        has_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters()
        )
        if has_nan:
            optimizer.zero_grad(set_to_none=True)
            continue

        nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
        optimizer.step()

        total_mae += loss.item()
        count += 1

    return total_mae / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, mean, std, device):
    model.eval()
    mt = torch.tensor(mean, device=device, dtype=torch.float32)
    st = torch.tensor(std, device=device, dtype=torch.float32)

    maes, rmses, mapes = [], [], []
    sums = {"abs": 0.0, "sq": 0.0, "ape": 0.0, "cnt": 0.0}

    for x, y, tod in loader:
        x, y, tod = x.to(device), y.to(device), tod.to(device)
        out = model(x, tod)
        pred_r = out * st + mt
        y_r = y * st + mt

        if cfg.IS_GRID:
            err = pred_r - y_r
            mask = metric_mask(y_r)
            sums["abs"] += (err.abs() * mask).sum().item()
            sums["sq"] += ((err ** 2) * mask).sum().item()
            sums["ape"] += ((err.abs() / (y_r.abs() + 1e-8)) * mask).sum().item()
            sums["cnt"] += mask.sum().item()
        else:
            maes.append(masked_mae(pred_r, y_r).item())
            rmses.append(masked_rmse(pred_r, y_r).item())
            mapes.append(masked_mape(pred_r, y_r).item())

    if cfg.IS_GRID:
        cnt = max(sums["cnt"], 1.0)
        return (
            float(sums["abs"] / cnt),
            float(np.sqrt(sums["sq"] / cnt)),
            float(100.0 * sums["ape"] / cnt),
        )

    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(mapes))


@torch.no_grad()
def measure_inference_time(model, loader, device, warmup=5, repeat=20):
    model.eval()
    times = []
    used = 0

    for idx, (x, _, tod) in enumerate(loader):
        x, tod = x.to(device), tod.to(device)
        if idx < warmup:
            _ = model(x, tod)
            continue
        if used >= repeat:
            break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x, tod)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        used += 1

    return float(np.mean(times)) if times else float("nan")


def cfg_dict_for_save():
    return {
        "MODEL": "ERST",
        "DATASET_NAME": cfg.DATASET_NAME,
        "DATA_PATH": cfg.DATA_PATH,
        "SAVE_PREFIX": cfg.SAVE_PREFIX,
        "IN_FEAT": cfg.IN_FEAT,
        "FLOW_INDEX": cfg.FLOW_INDEX,
        "TARGET_NAME": cfg.TARGET_NAME,
        "TARGET_INDEX": cfg.TARGET_INDEX,
        "IS_GRID": cfg.IS_GRID,
        "NUM_NODES": cfg.NUM_NODES,
        "TRAIN_RATIO": cfg.TRAIN_RATIO,
        "VAL_RATIO": cfg.VAL_RATIO,
        "IN_LEN": cfg.IN_LEN,
        "OUT_LEN": cfg.OUT_LEN,
        "D_MODEL": cfg.D_MODEL,
        "NUM_LAYERS": cfg.NUM_LAYERS,
        "DROPOUT": cfg.DROPOUT,
        "NUM_TOD": cfg.NUM_TOD,
        "INTERVAL_TEXT": cfg.INTERVAL_TEXT,
        "METRIC_MASK_VAL": cfg.METRIC_MASK_VAL,
        "USE_LAST_RESIDUAL": cfg.USE_LAST_RESIDUAL,
        "INTERACTION": "F(H)-F(H_rep_i)",
        "REFERENCE": "psi([history_i, mean_other_entities, time_of_day, layer_embedding])",
        "REPLACEMENT": "replace only entity i representation; preserve entities/count/normalization/function",
        "ADAPTIVE_GRAPH": "OFF",
        "PAIRWISE_ATTENTION": "OFF",
        "LR": cfg.LR,
        "WEIGHT_DECAY": cfg.WEIGHT_DECAY,
        "SEED": cfg.SEED,
    }


def run_experiment():
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    update_save_prefix()

    hr("═")
    print("  ERST — Entity Replacement for Spatiotemporal Forecasting")
    print("  Full model only / graph-attention free")
    hr("═")
    print(f"  Dataset : {cfg.DATASET_NAME}")
    if cfg.IS_GRID:
        print(f"  Target  : {cfg.TARGET_NAME}")
        print(
            f"  Protocol: {cfg.IN_LEN}->{cfg.OUT_LEN}, split 7:1:2, "
            f"{cfg.INTERVAL_TEXT}, true >= {cfg.METRIC_MASK_VAL:g}"
        )
    else:
        print(
            f"  Protocol: {cfg.IN_LEN}->{cfg.OUT_LEN}, split "
            f"{cfg.TRAIN_RATIO:.1f}/{cfg.VAL_RATIO:.1f}/"
            f"{1-cfg.TRAIN_RATIO-cfg.VAL_RATIO:.1f}, {cfg.INTERVAL_TEXT}"
        )
    print(f"  Device  : {device}")
    print(f"  Data    : {cfg.DATA_PATH}")
    print(f"  Output  : {cfg.SAVE_DIR}")

    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    dl_tr, dl_val, dl_te, mean, std = load_data(cfg.DATA_PATH, cfg.BATCH_SIZE)

    model = ERST().to(device)
    params = count_params(model)
    print(f"\n  Parameters: {params:,}")
    for name, mod in [
        ("FlowProjection", model.flow_proj),
        ("TemporalEncoder", model.temporal_enc),
        ("EntityEmbedding", model.entity_emb),
        ("EntityFFNLayers", model.layers),
        ("ReplacementResponse", model.response_layer),
        ("Decoder", model.decoder),
    ]:
        n = count_params(mod)
        ratio = 100 * n / params if params > 0 else 0
        print(f"    {name:22s}: {n:>10,}  ({ratio:5.1f}%)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_mae = float("inf")
    best_state = None
    no_improve = 0

    hr("═")
    print(f"  {'Ep':>4} {'TrMAE':>8} {'VlMAE':>8} {'VlRMSE':>8} {'VlMAPE':>8}  {'LR':>9}")
    hr("═")

    for epoch in range(1, cfg.EPOCHS + 1):
        tr_mae = train_epoch(model, dl_tr, optimizer, mean, std, device)
        val_mae, val_rmse, val_mape = evaluate(model, dl_val, mean, std, device)
        scheduler.step(val_mae)
        lr_now = optimizer.param_groups[0]["lr"]

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            tag = " ★"
        else:
            no_improve += 1
            tag = ""

        print(
            f"  {epoch:>4d} {tr_mae:>8.4f} {val_mae:>8.4f} "
            f"{val_rmse:>8.4f} {val_mape:>7.2f}%  {lr_now:.2e}{tag}"
        )

        if no_improve >= cfg.PATIENCE:
            print(f"\n  [Early Stop] epoch={epoch}  best_val_MAE={best_val_mae:.4f}")
            break

    if best_state is None:
        raise RuntimeError("No valid best_state was saved.")

    model.load_state_dict(best_state)
    test_mae, test_rmse, test_mape = evaluate(model, dl_te, mean, std, device)
    inf_time = measure_inference_time(model, dl_te, device)

    hr("═")
    print("  [FINAL TEST RESULT]")
    print(f"  MAE={test_mae:.4f}  RMSE={test_rmse:.4f}  MAPE={test_mape:.2f}%")
    print(f"  params={params:,}  inference={inf_time:.2f}ms/batch")
    gpu_mem_stat("final")
    hr("═")

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    save_path = os.path.join(cfg.SAVE_DIR, f"{cfg.SAVE_PREFIX}_best.pt")
    result_path = os.path.join(cfg.SAVE_DIR, f"{cfg.SAVE_PREFIX}_result.json")

    result = {
        "dataset": cfg.DATASET_NAME,
        "model": "ERST",
        "mae": float(test_mae),
        "rmse": float(test_rmse),
        "mape": float(test_mape),
        "params": int(params),
        "inference_ms_per_batch": float(inf_time),
        "save_path": save_path,
        "cfg": cfg_dict_for_save(),
    }

    torch.save(
        {
            "model_state": best_state,
            "mean": float(mean),
            "std": float(std),
            "test": {
                "mae": float(test_mae),
                "rmse": float(test_rmse),
                "mape": float(test_mape),
                "inference_ms_per_batch": float(inf_time),
                "params": int(params),
            },
            "cfg": cfg_dict_for_save(),
        },
        save_path,
    )

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  [Saved checkpoint] {save_path}")
    print(f"  [Saved result]     {result_path}")
    return result


# ============================================================
# 6. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ERST full-model implementation for spatiotemporal forecasting."
    )
    parser.add_argument("--dataset", type=str, default="NYCTaxi", choices=ALL_DATASETS + ["TDrive"])
    parser.add_argument(
        "--run_all_grid",
        action="store_true",
        help="Run NYCTaxi-inflow, CHIBike-outflow, and T-Drive-inflow sequentially.",
    )
    parser.add_argument(
        "--run_all_node",
        action="store_true",
        help="Run every node-dataset preset sequentially using --data_root.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for preset dataset paths.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Optional path override for a single dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        choices=["auto", "inflow", "outflow"],
        help="Grid target. auto = CHIBike outflow; NYCTaxi/T-Drive inflow.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_root",
        type=str,
        default="./outputs",
        help="Root directory for checkpoints/results.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional output directory override. For multi-dataset runs, dataset subfolders are added.",
    )
    parser.add_argument("--in_len", type=int, default=None)
    parser.add_argument("--out_len", type=int, default=None)
    parser.add_argument(
        "--metric_mask_val",
        type=float,
        default=-1.0,
        help="Optional grid metric/loss mask override; preset defaults are recommended.",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[argparse] ignored unknown args: {unknown}")
    return args


def _apply_runtime_args(args, dataset_name, is_multi_run):
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.NUM_WORKERS = args.num_workers
    cfg.D_MODEL = args.d_model
    cfg.NUM_LAYERS = args.num_layers
    cfg.DROPOUT = args.dropout
    cfg.LR = args.lr
    cfg.WEIGHT_DECAY = args.weight_decay
    cfg.PATIENCE = args.patience
    cfg.SEED = args.seed

    if args.save_dir is not None:
        cfg.SAVE_DIR = (
            str(Path(args.save_dir) / dataset_name)
            if is_multi_run
            else args.save_dir
        )

    if args.in_len is not None:
        if cfg.IS_GRID and args.in_len != 6:
            raise ValueError("Grid benchmark comparison requires --in_len 6.")
        cfg.IN_LEN = args.in_len
    if args.out_len is not None:
        if cfg.IS_GRID and args.out_len != 1:
            raise ValueError("Grid benchmark comparison requires --out_len 1.")
        cfg.OUT_LEN = args.out_len
    if args.metric_mask_val >= 0:
        cfg.METRIC_MASK_VAL = float(args.metric_mask_val)

    update_save_prefix()


if __name__ == "__main__":
    args = parse_args()
    if args.run_all_grid and args.run_all_node:
        raise ValueError("Choose only one of --run_all_grid or --run_all_node.")

    cfg.DATA_ROOT = args.data_root
    cfg.OUTPUT_ROOT = args.output_root

    if args.run_all_grid:
        dataset_names = GRID_DATASETS
    elif args.run_all_node:
        dataset_names = NODE_DATASETS
    else:
        dataset_names = [args.dataset]

    is_multi_run = len(dataset_names) > 1
    all_results = {}

    for dataset_name in dataset_names:
        selected_path = "" if is_multi_run else args.data_path
        selected_target = "auto" if is_multi_run else args.target
        set_dataset(dataset_name, data_path=selected_path, target=selected_target)
        _apply_runtime_args(args, dataset_name, is_multi_run)

        result = run_experiment()
        all_results[dataset_name] = result

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(all_results if is_multi_run else result)
