import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -------------------------------------------------------------------------
# Tensor notation
# -------------------------------------------------------------------------
# B: batch size
# S: number of sensors
# S_max: maximum number of sensors in a batch
# L: byte sequence length
# M: metadata dimension

# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class PrefixDataset(Dataset):
    """
    Build dataset using sensor observation prefixes.

    Each event detected by n sensors generates n samples:
        [s1], [s1, s2], [s1, s2, s3], ..., [s1, ..., sn]
    """

    def __init__(
        self,
        events_csv: str,
        observations_csv: str,
        event_ids: Optional[Sequence[int]] = None,
        normalize_metadata: bool = True,
        coord_scale: Optional[float] = None,
        time_scale: Optional[float] = None,
        min_sensors: int = 1,
        expand_prefixes: bool = True,
        save_stats_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.events_csv = str(events_csv)
        self.observations_csv = str(observations_csv)
        self.normalize_metadata = normalize_metadata
        self.min_sensors = int(min_sensors)
        self.expand_prefixes = bool(expand_prefixes)

        self.events_df = pd.read_csv(self.events_csv)
        self.obs_df = pd.read_csv(self.observations_csv)

        self.target_cols = [f"target_b{i}" for i in range(128)]
        self.rx_cols = [f"rx_b{i}" for i in range(128)]

        required_event_cols = ["event_id", *self.target_cols]
        required_obs_cols = ["event_id", "delta_arrival_time", *self.rx_cols]

        _check_required_columns(self.events_df, required_event_cols, "events.csv")
        _check_required_columns(self.obs_df, required_obs_cols, "observations.csv")

        self.meta_cols = ["delta_arrival_time"]

        self.events_df["event_id"] = self.events_df["event_id"].astype(np.int64)
        self.obs_df["event_id"] = self.obs_df["event_id"].astype(np.int64)

        self.obs_df = self.obs_df.sort_values(["event_id"]).reset_index(drop=True)

        counts = self.obs_df.groupby("event_id").size()
        valid_event_ids = set(counts[counts >= self.min_sensors].index.to_numpy(dtype=np.int64).tolist())

        if event_ids is not None:
            requested_event_ids = set(int(event_id) for event_id in event_ids)
            valid_event_ids = valid_event_ids.intersection(requested_event_ids)

        valid_event_ids = sorted(valid_event_ids)

        self.events_df = self.events_df[self.events_df["event_id"].isin(valid_event_ids)].copy()
        self.events_df = self.events_df.sort_values("event_id").reset_index(drop=True)

        self.obs_df = self.obs_df[self.obs_df["event_id"].isin(valid_event_ids)].copy()
        self.obs_df = self.obs_df.reset_index(drop=True)

        self.obs_grouped = self.obs_df.groupby("event_id", sort=False)
        self.event_ids = self.events_df["event_id"].to_numpy(dtype=np.int64)

        self.event_row_by_id = {}
        for _, row in self.events_df.iterrows():
            event_id = int(row["event_id"])
            self.event_row_by_id[event_id] = row

        if coord_scale is None:
            coord_candidates = []

            for col in ("sensor_x", "sensor_y"):
                if col in self.obs_df.columns and len(self.obs_df) > 0:
                    coord_candidates.append(float(self.obs_df[col].max()))

            if "event_x" in self.events_df.columns and len(self.events_df) > 0:
                coord_candidates.append(float(self.events_df["event_x"].max()))

            if "event_y" in self.events_df.columns and len(self.events_df) > 0:
                coord_candidates.append(float(self.events_df["event_y"].max()))

            coord_scale = max(coord_candidates) if coord_candidates else 1.0

        if time_scale is None:
            if len(self.obs_df) > 0:
                time_scale = float(self.obs_df["delta_arrival_time"].max())
            else:
                time_scale = 1.0

        self.coord_scale = max(float(coord_scale), 1e-8)
        self.time_scale = max(float(time_scale), 1e-8)

        self.samples: List[Tuple[int, int]] = []

        for event_id in self.event_ids:
            num_sensors = len(self.obs_grouped.get_group(int(event_id)))

            if self.expand_prefixes:
                for prefix_len in range(1, num_sensors + 1):
                    self.samples.append((int(event_id), prefix_len))
            else:
                self.samples.append((int(event_id), num_sensors))

        if save_stats_path is not None:
            stats = {
                "coord_scale": self.coord_scale,
                "time_scale": self.time_scale,
                "meta_cols": self.meta_cols,
                "normalize_metadata": self.normalize_metadata,
                "min_sensors": self.min_sensors,
                "expand_prefixes": self.expand_prefixes,
                "num_events_after_filter": int(len(self.events_df)),
                "num_observations_after_filter": int(len(self.obs_df)),
                "num_samples_after_prefix_expansion": int(len(self.samples)),
            }

            Path(save_stats_path).write_text(
                json.dumps(stats, indent=2),
                encoding="utf-8",
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        event_id, prefix_len = self.samples[idx]

        event_row = self.event_row_by_id[event_id]
        obs_rows = self.obs_grouped.get_group(event_id).iloc[:prefix_len]

        sensor_bytes = obs_rows[self.rx_cols].to_numpy(dtype=np.int64)     # (S, L)
        sensor_meta = obs_rows[self.meta_cols].to_numpy(dtype=np.float32)  # (S, M)

        if self.normalize_metadata:
            sensor_meta = sensor_meta.copy()
            sensor_meta[:, 0] /= self.time_scale

        target_bytes = event_row[self.target_cols].to_numpy(dtype=np.int64)  # (L,)

        return (
            torch.tensor(sensor_bytes, dtype=torch.long),
            torch.tensor(sensor_meta, dtype=torch.float32),
            torch.tensor(target_bytes, dtype=torch.long),
            torch.tensor(event_id, dtype=torch.long),
        )

    @property
    def meta_dim(self) -> int:
        return len(self.meta_cols)

    def summary(self) -> Dict:
        obs_counts = self.obs_df.groupby("event_id").size().to_numpy()

        prefix_lengths = np.array(
            [prefix_len for _, prefix_len in self.samples],
            dtype=np.int64,
        )

        return {
            "num_events": int(len(self.events_df)),
            "num_observations": int(len(self.obs_df)),
            "num_samples": int(len(self.samples)),
            "expand_prefixes": bool(self.expand_prefixes),
            "meta_cols": list(self.meta_cols),
            "meta_dim": int(self.meta_dim),
            "normalize_metadata": bool(self.normalize_metadata),
            "coord_scale": float(self.coord_scale),
            "time_scale": float(self.time_scale),
            "min_sensors_per_event": int(self.min_sensors),
            "min_detected_sensors": int(obs_counts.min()) if len(obs_counts) > 0 else 0,
            "max_detected_sensors": int(obs_counts.max()) if len(obs_counts) > 0 else 0,
            "mean_detected_sensors": float(obs_counts.mean()) if len(obs_counts) > 0 else 0.0,
            "min_prefix_len": int(prefix_lengths.min()) if len(prefix_lengths) > 0 else 0,
            "max_prefix_len": int(prefix_lengths.max()) if len(prefix_lengths) > 0 else 0,
            "mean_prefix_len": float(prefix_lengths.mean()) if len(prefix_lengths) > 0 else 0.0,
        }

def collate(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
):

    if len(batch) == 0:
        raise ValueError("The batch is empty.")

    max_sensors = max(sensor_bytes.shape[0] for sensor_bytes, _, _, _ in batch)
    byte_len = batch[0][0].shape[1]
    meta_dim = batch[0][1].shape[1]

    sensor_bytes_batch = []
    sensor_meta_batch = []
    sensor_mask_batch = []
    target_batch = []
    event_id_batch = []

    for sensor_bytes, sensor_meta, target, event_id in batch:
        num_sensors = sensor_bytes.shape[0]
        pad_sensors = max_sensors - num_sensors

        padded_bytes = torch.cat([sensor_bytes, torch.zeros(pad_sensors, byte_len, dtype=torch.long)], dim=0)

        padded_meta = torch.cat([sensor_meta, torch.zeros(pad_sensors, meta_dim, dtype=torch.float32)], dim=0)

        sensor_mask = torch.cat([torch.ones(num_sensors, dtype=torch.float32), torch.zeros(pad_sensors, dtype=torch.float32)], dim=0)

        sensor_bytes_batch.append(padded_bytes)
        sensor_meta_batch.append(padded_meta)
        sensor_mask_batch.append(sensor_mask)
        target_batch.append(target)
        event_id_batch.append(event_id)

    return (
        torch.stack(sensor_bytes_batch), # (B, S_max, L)
        torch.stack(sensor_meta_batch), # (B, S_max, M)
        torch.stack(sensor_mask_batch), # (B, S_max)
        torch.stack(target_batch), # (B, L)
        torch.stack(event_id_batch), # (B,)
    )

# -------------------------------------------------------------------------
# Train/validation/test split
# -------------------------------------------------------------------------

def split_event_ids(
    events_csv: str,
    observations_csv: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_sensors: int = 1,
    seed: int = 1234,
) -> Dict[str, List[int]]:

    events_df = pd.read_csv(events_csv)
    obs_df = pd.read_csv(observations_csv)

    events_df["event_id"] = events_df["event_id"].astype(np.int64)
    obs_df["event_id"] = obs_df["event_id"].astype(np.int64)

    counts = obs_df.groupby("event_id").size()
    valid_event_ids = counts[counts >= int(min_sensors)].index.to_numpy(dtype=np.int64)

    valid_event_ids = np.array(sorted(set(valid_event_ids).intersection(set(events_df["event_id"].tolist()))), dtype=np.int64)

    rng = np.random.default_rng(seed)
    rng.shuffle(valid_event_ids)

    num_events = len(valid_event_ids)
    num_train = int(num_events * train_ratio)
    num_val = int(num_events * val_ratio)

    train_ids = valid_event_ids[:num_train].tolist()
    val_ids = valid_event_ids[num_train:num_train + num_val].tolist()
    test_ids = valid_event_ids[num_train + num_val:].tolist()

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

def build_datasets_from_split(
    events_csv: str,
    observations_csv: str,
    split_dict: Dict[str, Sequence[int]],
    normalize_metadata: bool = True,
    coord_scale: Optional[float] = None,
    time_scale: Optional[float] = None,
    min_sensors: int = 1,
    expand_prefixes: bool = True,
):

    train_dataset = PrefixDataset(
        events_csv=events_csv,
        observations_csv=observations_csv,
        event_ids=split_dict["train"],
        normalize_metadata=normalize_metadata,
        coord_scale=coord_scale,
        time_scale=time_scale,
        min_sensors=min_sensors,
        expand_prefixes=expand_prefixes,
    )

    if coord_scale is None:
        coord_scale = train_dataset.coord_scale

    if time_scale is None:
        time_scale = train_dataset.time_scale

    val_dataset = PrefixDataset(
        events_csv=events_csv,
        observations_csv=observations_csv,
        event_ids=split_dict["val"],
        normalize_metadata=normalize_metadata,
        coord_scale=coord_scale,
        time_scale=time_scale,
        min_sensors=min_sensors,
        expand_prefixes=expand_prefixes,
    )

    test_dataset = PrefixDataset(
        events_csv=events_csv,
        observations_csv=observations_csv,
        event_ids=split_dict["test"],
        normalize_metadata=normalize_metadata,
        coord_scale=coord_scale,
        time_scale=time_scale,
        min_sensors=min_sensors,
        expand_prefixes=expand_prefixes,
    )

    return train_dataset, val_dataset, test_dataset

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def _check_required_columns(
    df: pd.DataFrame,
    required_cols: Sequence[str],
    name: str,
) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in {name}: {missing_cols}")