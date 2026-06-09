import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_datasets_from_split, collate, split_event_ids
from transformer_fusion_model import ByteReconstructionModel

# -------------------------------------------------------------------------
# Mask Saving configuration
# -------------------------------------------------------------------------

BYTE_CONFIDENCE_THRESHOLD = 0.8

# -------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------

def build_full_byte_mask(sensor_bytes: torch.Tensor) -> torch.Tensor:
    return torch.ones(sensor_bytes.shape, dtype=torch.float32, device=sensor_bytes.device)

# -------------------------------------------------------------------------
# Training and normal evaluation
# -------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for sensor_bytes, sensor_meta, sensor_mask, target, event_ids in dataloader:
        sensor_bytes = sensor_bytes.to(device)
        sensor_meta = sensor_meta.to(device)
        sensor_mask = sensor_mask.to(device)
        target = target.to(device)

        sensor_byte_mask = build_full_byte_mask(sensor_bytes)

        optimizer.zero_grad()

        logits = model(sensor_bytes, sensor_meta, sensor_mask, sensor_byte_mask)

        loss = criterion(logits.reshape(-1, 256), target.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_bytes = 0
    correct_bytes = 0
    total_sequences = 0
    correct_sequences = 0

    for sensor_bytes, sensor_meta, sensor_mask, target, event_ids in dataloader:
        sensor_bytes = sensor_bytes.to(device)
        sensor_meta = sensor_meta.to(device)
        sensor_mask = sensor_mask.to(device)
        target = target.to(device)

        sensor_byte_mask = build_full_byte_mask(sensor_bytes)
        logits = model(sensor_bytes, sensor_meta, sensor_mask, sensor_byte_mask)

        loss = criterion(logits.reshape(-1, 256), target.reshape(-1))
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)

        correct_bytes += (preds == target).sum().item()
        total_bytes += target.numel()

        seq_correct = (preds == target).all(dim=1)
        correct_sequences += seq_correct.sum().item()
        total_sequences += target.shape[0]

    return {
        "loss": total_loss / len(dataloader),
        "byte_accuracy": correct_bytes / total_bytes if total_bytes > 0 else 0.0,
        "sequence_accuracy": correct_sequences / total_sequences if total_sequences > 0 else 0.0,
    }

# -------------------------------------------------------------------------
# Mask evaluation
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mask(model, dataloader, device):
    model.eval()

    total_bytes = 0
    correct_bytes = 0
    total_sequences = 0
    correct_sequences = 0

    transmitted_bytes = 0
    max_possible_bytes = 0

    for sensor_bytes, sensor_meta, sensor_mask, target, event_ids in dataloader:
        sensor_bytes = sensor_bytes.to(device)
        sensor_meta = sensor_meta.to(device)
        sensor_mask = sensor_mask.to(device)
        target = target.to(device)

        B, S, L = sensor_bytes.shape

        byte_mask = torch.zeros(B, S, L, device=device)
        byte_mask[:, 0, :] = 1.0

        valid_sensors = sensor_mask.sum(dim=1).long()
        final_preds = None

        for s in range(S):
            active = valid_sensors > s

            if not active.any():
                continue

            logits = model(sensor_bytes, sensor_meta, sensor_mask, byte_mask)
            probs = torch.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

            final_preds = preds
            next_s = s + 1

            if next_s < S:
                next_exists = valid_sensors > next_s

                if next_exists.any():
                    if next_s == 1:
                        byte_mask[next_exists, next_s, :] = 1.0
                    else:
                        accepted = conf >= BYTE_CONFIDENCE_THRESHOLD
                        byte_mask[next_exists, next_s, :] = 1.0 - accepted[next_exists].float()

        preds = final_preds

        correct_bytes += (preds == target).sum().item()
        total_bytes += target.numel()

        seq_correct = (preds == target).all(dim=1)
        correct_sequences += seq_correct.sum().item()
        total_sequences += target.shape[0]

        valid_sensor_byte_mask = sensor_mask.unsqueeze(-1).expand_as(byte_mask)
        transmitted_bytes += (byte_mask * valid_sensor_byte_mask).sum().item()
        max_possible_bytes += (sensor_mask.sum(dim=1) * L).sum().item()

    saving_ratio = 1.0 - transmitted_bytes / max_possible_bytes if max_possible_bytes > 0 else 0.0

    return {
        "byte_accuracy": correct_bytes / total_bytes if total_bytes > 0 else 0.0,
        "sequence_accuracy": correct_sequences / total_sequences if total_sequences > 0 else 0.0,
        "transmitted_bytes": int(transmitted_bytes),
        "max_possible_bytes": int(max_possible_bytes),
        "saving_ratio": float(saving_ratio),
    }

# -------------------------------------------------------------------------
# Mask evaluation export
# -------------------------------------------------------------------------

@torch.no_grad()
def save_mask_predictions(model, dataloader, device, output_json="test_predictions_mask.json"):
    model.eval()
    rows = []

    for sensor_bytes, sensor_meta, sensor_mask, target, event_ids in dataloader:
        sensor_bytes = sensor_bytes.to(device)
        sensor_meta = sensor_meta.to(device)
        sensor_mask = sensor_mask.to(device)
        target = target.to(device)

        B, S, L = sensor_bytes.shape

        byte_mask = torch.zeros(B, S, L, device=device)
        byte_mask[:, 0, :] = 1.0

        valid_sensors = sensor_mask.sum(dim=1).long()
        prev_preds = None

        differential_saved_bytes = torch.zeros(B, dtype=torch.long, device=device)
        differential_masked_saved_bytes = torch.zeros(B, dtype=torch.long, device=device)

        for s in range(S):
            active = valid_sensors > s

            if not active.any():
                continue

            logits = model(sensor_bytes, sensor_meta, sensor_mask, byte_mask)
            probs = torch.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)

            effective_byte_mask = byte_mask * sensor_mask.unsqueeze(-1).expand_as(byte_mask)

            for i in range(B):
                if not active[i]:
                    continue

                event_id_i = int(event_ids[i].item())
                num_sensors_step = s + 1

                target_i = target[i].detach().cpu().numpy()
                pred_i = preds[i].detach().cpu().numpy()
                conf_i = conf[i].detach().cpu().numpy()

                sensor_bytes_i = sensor_bytes[i, :num_sensors_step].detach().cpu().numpy()
                sensor_meta_i = sensor_meta[i, :num_sensors_step].detach().cpu().numpy()
                byte_mask_i = effective_byte_mask[i, :num_sensors_step].detach().cpu().numpy().astype(int)

                transmitted_per_sensor = byte_mask_i.sum(axis=1).astype(int).tolist()
                total_transmitted = int(byte_mask_i.sum())
                max_possible = int(num_sensors_step * L)
                saving_ratio = 1.0 - total_transmitted / max_possible if max_possible > 0 else 0.0

                num_correct = int((target_i == pred_i).sum())
                byte_acc = float((target_i == pred_i).mean())
                seq_acc = int(np.array_equal(target_i, pred_i))

                accepted_i = conf_i >= BYTE_CONFIDENCE_THRESHOLD
                num_accepted_bytes = int(accepted_i.sum())
                num_lost_bytes = int(((pred_i != target_i) & accepted_i).sum())
                accepted_error_rate = num_lost_bytes / num_accepted_bytes if num_accepted_bytes > 0 else 0.0

                if prev_preds is not None:
                    prev_pred_i = prev_preds[i].detach().cpu().numpy()
                    obs_vs_prev_pred = sensor_bytes_i[s] == prev_pred_i
                    masked_obs_vs_prev_pred = obs_vs_prev_pred[byte_mask_i[s] == 1]

                    differential_saved_bytes[i] += int(obs_vs_prev_pred.sum())
                    differential_masked_saved_bytes[i] += int(masked_obs_vs_prev_pred.sum())

                rows.append({
                    "event_id": event_id_i,
                    "num_sensors": int(num_sensors_step),
                    "target": target_i.tolist(),
                    "prediction": pred_i.tolist(),
                    "byte_confidence": conf_i.tolist(),
                    "mean_byte_confidence": float(conf_i.mean()),
                    "sensor_bytes_original": sensor_bytes_i.tolist(),
                    "sensor_meta": sensor_meta_i.tolist(),
                    "sensor_byte_mask": byte_mask_i.tolist(),
                    "masked_bytes_per_sensor": transmitted_per_sensor,
                    "total_masked_bytes": total_transmitted,
                    "max_possible_bytes": max_possible,
                    "saving_ratio": float(saving_ratio),
                    "num_correct_bytes": num_correct,
                    "num_error_bytes": int(L - num_correct),
                    "byte_accuracy": byte_acc,
                    "sequence_correct": seq_acc,
                    "num_accepted_bytes": num_accepted_bytes,
                    "num_lost_bytes": num_lost_bytes,
                    "accepted_error_rate": float(accepted_error_rate),
                    "differential_saved_bytes": int(differential_saved_bytes[i].item()),
                    "differential_masked_saved_bytes": int(differential_masked_saved_bytes[i].item()),
                })

            next_s = s + 1

            if next_s < S:
                next_exists = valid_sensors > next_s

                if next_exists.any():
                    if next_s == 1:
                        byte_mask[next_exists, next_s, :] = 1.0
                    else:
                        accepted = conf >= BYTE_CONFIDENCE_THRESHOLD
                        byte_mask[next_exists, next_s, :] = 1.0 - accepted[next_exists].float()

            prev_preds = preds.clone()

    rows.sort(key=lambda row: (row["event_id"], row["num_sensors"]))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Mask predictions saved to: {output_json}")

def save_sensor_metrics_mask(
    predictions_json="test_predictions_mask.json",
    output_csv="metrics_by_num_sensors_mask.csv",
):
    df = pd.read_json(predictions_json)

    grouped = df.groupby("num_sensors").agg(
        mean_byte_accuracy=("byte_accuracy", "mean"),
        mean_error_bytes=("num_error_bytes", "mean"),
        sequence_accuracy=("sequence_correct", "mean"),
        num_samples=("num_sensors", "size"),
        mean_confidence=("mean_byte_confidence", "mean"),
        mean_saving_ratio=("saving_ratio", "mean"),
        mean_masked_bytes=("total_masked_bytes", "mean"),
        mean_max_possible_bytes=("max_possible_bytes", "mean"),
        mean_accepted_bytes=("num_accepted_bytes", "mean"),
        total_accepted_bytes=("num_accepted_bytes", "sum"),
        mean_lost_bytes=("num_lost_bytes", "mean"),
        total_lost_bytes=("num_lost_bytes", "sum"),
        mean_accepted_error_rate=("accepted_error_rate", "mean"),
    ).reset_index()

    grouped.to_csv(output_csv, index=False)

    print(f"Mask metrics by number of sensors saved to: {output_csv}")
    print(grouped)

    return grouped

# -------------------------------------------------------------------------
# Normal evaluation export
# -------------------------------------------------------------------------

@torch.no_grad()
def save_normal_predictions_by_sensors(
    model,
    dataloader,
    device,
    output_json="test_predictions_normal_by_sensors.json",
):
    model.eval()
    rows = []

    for sensor_bytes, sensor_meta, sensor_mask, target, event_ids in dataloader:
        sensor_bytes = sensor_bytes.to(device)
        sensor_meta = sensor_meta.to(device)
        sensor_mask = sensor_mask.to(device)
        target = target.to(device)

        sensor_byte_mask = build_full_byte_mask(sensor_bytes)
        logits = model(sensor_bytes, sensor_meta, sensor_mask, sensor_byte_mask)

        probs = torch.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)

        B, S, L = sensor_bytes.shape

        for i in range(B):
            valid_sensor_mask_i = sensor_mask[i].bool()

            event_id_i = int(event_ids[i].item())
            num_sensors = int(valid_sensor_mask_i.sum().item())

            target_i = target[i].detach().cpu().numpy()
            pred_i = preds[i].detach().cpu().numpy()
            conf_i = conf[i].detach().cpu().numpy()

            sensor_bytes_i = sensor_bytes[i][valid_sensor_mask_i].detach().cpu().numpy()
            sensor_meta_i = sensor_meta[i][valid_sensor_mask_i].detach().cpu().numpy()

            num_correct = int((target_i == pred_i).sum())
            num_errors = int(L - num_correct)
            byte_acc = float((target_i == pred_i).mean())
            seq_acc = int(np.array_equal(target_i, pred_i))

            rows.append({
                "event_id": event_id_i,
                "num_sensors": num_sensors,
                "target": target_i.tolist(),
                "prediction": pred_i.tolist(),
                "byte_confidence": conf_i.tolist(),
                "mean_byte_confidence": float(conf_i.mean()),
                "sensor_bytes": sensor_bytes_i.tolist(),
                "sensor_meta": sensor_meta_i.tolist(),
                "num_correct_bytes": num_correct,
                "num_error_bytes": num_errors,
                "byte_accuracy": byte_acc,
                "sequence_correct": seq_acc,
            })

    rows.sort(key=lambda row: (row["event_id"], row["num_sensors"]))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Normal predictions by number of sensors saved to: {output_json}")


def save_sensor_metrics_normal(
    predictions_json="test_predictions_normal_by_sensors.json",
    output_csv="metrics_by_num_sensors_normal.csv",
):
    df = pd.read_json(predictions_json)

    grouped = df.groupby("num_sensors").agg(
        mean_byte_accuracy=("byte_accuracy", "mean"),
        mean_error_bytes=("num_error_bytes", "mean"),
        sequence_accuracy=("sequence_correct", "mean"),
        num_samples=("num_sensors", "size"),
        mean_confidence=("mean_byte_confidence", "mean"),
    ).reset_index()

    grouped.to_csv(output_csv, index=False)

    print(f"Normal metrics by number of sensors saved to: {output_csv}")
    print(grouped)

    return grouped

# -------------------------------------------------------------------------
# Full training
# -------------------------------------------------------------------------

def train(model, train_loader, val_loader, epochs, lr, device):
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_byte_accuracy": val_metrics["byte_accuracy"],
            "val_sequence_accuracy": val_metrics["sequence_accuracy"],
        }

        history.append(row)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Byte {val_metrics['byte_accuracy']:.4f} | "
            f"Val Seq {val_metrics['sequence_accuracy']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), "best_model.pt")
            print("  -> Best model saved")

    history_df = pd.DataFrame(history)
    history_df.to_csv("training_history.csv", index=False)

    return history_df

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    EVENTS_CSV = "events.csv"
    OBSERVATIONS_CSV = "observations.csv"

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    BATCH_SIZE = 8
    EPOCHS = 10
    LR = 1e-3
    SEED = 1234

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    split_dict = split_event_ids(
        EVENTS_CSV,
        OBSERVATIONS_CSV,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        min_sensors=1,
        seed=SEED,
    )

    train_dataset, val_dataset, _ = build_datasets_from_split(
        EVENTS_CSV,
        OBSERVATIONS_CSV,
        split_dict,
        expand_prefixes=True,
    )

    _, _, test_dataset = build_datasets_from_split(
        EVENTS_CSV,
        OBSERVATIONS_CSV,
        split_dict,
        expand_prefixes=False,
    )

    _, _, test_prefix_dataset = build_datasets_from_split(
        EVENTS_CSV,
        OBSERVATIONS_CSV,
        split_dict,
        expand_prefixes=True,
    )

    print("\nTRAIN SUMMARY")
    print(train_dataset.summary())

    print("\nVAL SUMMARY")
    print(val_dataset.summary())

    print("\nTEST FULL SUMMARY")
    print(test_dataset.summary())

    print("\nTEST PREFIX SUMMARY")
    print(test_prefix_dataset.summary())

    dataset_info = {
        "train_summary": train_dataset.summary(),
        "val_summary": val_dataset.summary(),
        "test_full_summary": test_dataset.summary(),
        "test_prefix_summary": test_prefix_dataset.summary(),
        "train_event_ids": len(split_dict["train"]),
        "val_event_ids": len(split_dict["val"]),
        "test_event_ids": len(split_dict["test"]),
        "confidence_threshold": BYTE_CONFIDENCE_THRESHOLD,
    }

    Path("dataset_split_summary.json").write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_prefix_loader = DataLoader(test_prefix_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = ByteReconstructionModel(
        hidden_dim=64,
        byte_emb_dim=16,
        meta_dim=1,
        num_heads=4,
        inter_num_layers=1,
    )

    train(model, train_loader, val_loader, EPOCHS, LR, device)

    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    normal_metrics = evaluate(model, test_loader, criterion, device)
    mask_metrics = evaluate_mask(model, test_loader, device)

    save_mask_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        output_json="test_predictions_mask.json",
    )

    save_sensor_metrics_mask(
        predictions_json="test_predictions_mask.json",
        output_csv="metrics_by_num_sensors_mask.csv",
    )

    save_normal_predictions_by_sensors(
        model=model,
        dataloader=test_prefix_loader,
        device=device,
        output_json="test_predictions_normal_by_sensors.json",
    )

    save_sensor_metrics_normal(
        predictions_json="test_predictions_normal_by_sensors.json",
        output_csv="metrics_by_num_sensors_normal.csv",
    )

    final = {
        "standard_test_full_events": normal_metrics,
        "mask_protocol": mask_metrics,
        "confidence_threshold": BYTE_CONFIDENCE_THRESHOLD,
        "files": {
            "mask_predictions": "test_predictions_mask.json",
            "mask_metrics_by_sensors": "metrics_by_num_sensors_mask.csv",
            "normal_predictions_by_sensors": "test_predictions_normal_by_sensors.json",
            "normal_metrics_by_sensors": "metrics_by_num_sensors_normal.csv",
        },
    }

    print("\nFINAL METRICS")
    print(json.dumps(final, indent=2))

    Path("final_test_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")