import sys, os
os.chdir(sys.path[0])
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pandas as pd
import matplotlib.pyplot as plt
import random
import yaml
import time
from Networks.Networks import PHOENIXUNet, CNNEmbed, OnlyUNet, OnlyCNN

# -----------------------------
# utils
# -----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_config(path='config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def worker_init_fn(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# -----------------------------
# dataset
# -----------------------------
class GasFieldDatasetPre(Dataset):
    """Dataset for preprocessed .npz files: expects keys [img, meta, label]."""
    def __init__(self, npz_files):
        self.files = npz_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        img = torch.from_numpy(data['img'])                # (3, H, W)
        meta = torch.from_numpy(data['meta'])              # (meta_dim,)
        label = torch.from_numpy(data['label']).unsqueeze(0)  # (1, H, W)
        return img, meta, label


def read_file_list(txt_path, npz_dir=None, add_prefix=False):
    with open(txt_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    if add_prefix:
        files = [f"pre_{fn}" if not fn.startswith("pre_") else fn for fn in files]
    if npz_dir is not None:
        files = [os.path.join(npz_dir, fn) for fn in files]
    return files

# -----------------------------
# loss & metrics
# -----------------------------
def get_loss_fn(name="mse"):
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name in ("huber", "smoothl1"):
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss: {name}")

def compute_metrics(pred, target, mask=None, thres=1e-4):
    """
    Basic metrics on (pred, target):
    - MAE, MSE, RMSE
    - IoU (binary > thres)
    - R2
    If mask is provided, apply element-wise before computing.
    """
    with torch.no_grad():
        if mask is not None:
            pred = pred * mask
            target = target * mask

        pred_np = pred.detach().cpu().numpy().ravel()
        target_np = target.detach().cpu().numpy().ravel()

        mae = np.mean(np.abs(pred_np - target_np))
        mse = np.mean((pred_np - target_np) ** 2)
        rmse = np.sqrt(mse)

        pred_bin = (pred_np > thres).astype(np.float32)
        target_bin = (target_np > thres).astype(np.float32)
        intersection = np.sum(pred_bin * target_bin)
        union = np.sum((pred_bin + target_bin) > 0)
        iou = intersection / (union + 1e-8)

        if len(target_np) > 1:
            ss_res = np.sum((target_np - pred_np) ** 2)
            ss_tot = np.sum((target_np - target_np.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)
        else:
            r2 = np.nan

        return {"mae": mae, "mse": mse, "rmse": rmse, "iou": iou, "r2": r2}

# -----------------------------
# train / eval
# -----------------------------
def train_epoch(model, dataloader, optimizer, device,
                loss_type="mse", epoch=0,
                calc_metrics=False, save_pred_once=False,
                out_dir='./'):
    """
    One training epoch.
    - calc_metrics: if True, compute metrics (slower), otherwise only loss.
    - save_pred_once: if True, save preds/labels of this epoch (e.g. for analysis).
    """
    model.train()
    total_loss = 0.0
    all_metrics = []
    preds_list, labels_list = [], []
    loss_fn = get_loss_fn(loss_type)

    for batch in dataloader:
        if batch is None:
            continue
        img, meta, label = batch
        img, meta, label = img.to(device), meta.to(device), label.to(device)

        pred = model(img, meta)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)

        if calc_metrics:
            metrics = compute_metrics(pred, label)
            all_metrics.append(metrics)

        if save_pred_once:
            preds_list.append(pred.detach().cpu().numpy())
            labels_list.append(label.detach().cpu().numpy())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)

    if calc_metrics and all_metrics:
        mean_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    else:
        mean_metrics = {}

    if save_pred_once and preds_list:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"train_pred_epoch{epoch:03d}.npz"),
            preds=np.concatenate(preds_list, axis=0),
            labels=np.concatenate(labels_list, axis=0)
        )

    return avg_loss, mean_metrics


@torch.no_grad()
def eval_epoch(model, dataloader, device,
               loss_type="mse", epoch=0,
               calc_metrics=True, save_pred_once=False,
               out_dir='./'):
    """
    One validation epoch.
    - calc_metrics: if True, compute metrics.
    - save_pred_once: if True, save preds/labels of this epoch.
    """
    model.eval()
    total_loss = 0.0
    all_metrics = []
    preds_list, labels_list = [], []
    loss_fn = get_loss_fn(loss_type)

    for batch in dataloader:
        if batch is None:
            continue
        img, meta, label = batch
        img, meta, label = img.to(device), meta.to(device), label.to(device)

        pred = model(img, meta)
        loss = loss_fn(pred, label)
        total_loss += loss.item() * img.size(0)

        if calc_metrics:
            metrics = compute_metrics(pred, label)
            all_metrics.append(metrics)

        if save_pred_once:
            preds_list.append(pred.detach().cpu().numpy())
            labels_list.append(label.detach().cpu().numpy())

    avg_loss = total_loss / max(len(dataloader.dataset), 1)

    if calc_metrics and all_metrics:
        mean_metrics = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    else:
        mean_metrics = {}

    if save_pred_once and preds_list:
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(out_dir, f"val_pred_epoch{epoch:03d}.npz"),
            preds=np.concatenate(preds_list, axis=0),
            labels=np.concatenate(labels_list, axis=0)
        )

    return avg_loss, mean_metrics

# -----------------------------
# main
# -----------------------------
def main(cfg, gpu_id=None):
    # 1. basic settings
    set_seed(cfg['seed'])
    npz_train_dir = cfg['npz_train_dir']
    npz_val_dir = cfg['npz_val_dir']
    train_txt = cfg['train_txt']
    val_txt = cfg['val_txt']
    model_type = cfg['model_type'].lower()

    # training params
    batch_size = cfg['batch_size']
    num_epochs = cfg['num_epochs']
    lr = cfg['lr']
    loss_type = cfg['loss_type']

    # architecture params
    base_ch = cfg['base_ch']
    depth = cfg['depth']
    fuse_dim = cfg['fuse_dim']

    # device
    if gpu_id is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('[INFO] Using device:', device)

    # 2. exp paths
    exp_name = f"FHENUNet_bs{batch_size}_lr{lr}_loss{loss_type}_ch{base_ch}_dim{fuse_dim}_depth{depth}_{model_type}"
    save_dir = f'./results/exps/{exp_name}'
    model_dir = os.path.join(save_dir, 'model')
    log_dir = os.path.join(save_dir, 'log')
    pred_dir = os.path.join(save_dir, 'pred_npz')
    train_pred_dir = os.path.join(pred_dir, 'train_pred_npz')
    val_pred_dir = os.path.join(pred_dir, 'val_pred_npz')
    fig_dir = os.path.join(save_dir, 'figures')

    for d in [save_dir, model_dir, log_dir, pred_dir,
              train_pred_dir, val_pred_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[INFO] Experiment results will be saved to {save_dir}")

    # 3. his.txt
    his_path = os.path.join(save_dir, 'his.txt')
    with open(his_path, 'w') as f:
        f.write(f"exp_name: {exp_name}\n")
        f.write(f"model_type: {model_type}, base_ch: {base_ch}, depth: {depth}, fuse_dim: {fuse_dim}\n")
        f.write(f"batch_size: {batch_size}, num_epochs: {num_epochs}, lr: {lr}, loss_type: {loss_type}\n")

    # 4. dataset & dataloader (preprocessed npz)
    train_files = read_file_list(train_txt, npz_train_dir, add_prefix=True)
    val_files = read_file_list(val_txt, npz_val_dir, add_prefix=True)

    train_dataset = GasFieldDatasetPre(train_files)
    val_dataset = GasFieldDatasetPre(val_files)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=safe_collate, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2,
        collate_fn=safe_collate, worker_init_fn=worker_init_fn
    )

    # 5. model init
    if model_type == "phoenixunet":
        model = PHOENIXUNet(in_ch=3, out_ch=1,
                         base_ch=base_ch, fuse_dim=fuse_dim, depth=depth).to(device)
    elif model_type == "onlyunet":
        model = OnlyUNet(in_ch=3, out_ch=1,
                         base_ch=base_ch, depth=depth).to(device)
    elif model_type == "onlycnn":
        model = OnlyCNN(in_ch=3, out_ch=1,
                        base_ch=base_ch, depth=depth).to(device)
    elif model_type == "cnnembed":
        model = CNNEmbed(in_ch=3, out_ch=1,
                        base_ch=base_ch, fuse_dim=fuse_dim, depth=depth).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 6. optimizer & param log
    n_params = count_parameters(model)
    print(f"[INFO] Model parameter count: {n_params:,}")
    with open(his_path, 'a') as f:
        f.write(f"param_count: {n_params}\n")

    optimizer = Adam(model.parameters(), lr=lr)
    save_path_best = os.path.join(model_dir, 'best_model.pt')
    log_train_csv = os.path.join(log_dir, 'train_log.csv')
    log_val_csv = os.path.join(log_dir, 'val_log.csv')

    # 7. training loop
    best_val_loss = float('inf')
    train_log, val_log = [], []
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        save_pred_flag = (epoch % 50 == 0 or epoch == num_epochs)

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            loss_type=loss_type,
            epoch=epoch,
            calc_metrics=True,
            save_pred_once=save_pred_flag,
            out_dir=train_pred_dir
        )

        val_loss, val_metrics = eval_epoch(
            model, val_loader, device,
            loss_type=loss_type,
            epoch=epoch,
            calc_metrics=True, 
            save_pred_once=save_pred_flag,
            out_dir=val_pred_dir
        )

        epoch_time = time.time() - epoch_start

        t_iou = train_metrics.get('iou', float('nan'))
        v_iou = val_metrics.get('iou', float('nan'))
        t_mae = train_metrics.get('mae', float('nan'))
        v_mae = val_metrics.get('mae', float('nan'))
        t_r2 = train_metrics.get('r2', float('nan'))
        v_r2 = val_metrics.get('r2', float('nan'))

        print(f"Epoch {epoch:03d} | "
              f"Train Loss(mse): {train_loss:.6f} | Val Loss(mse): {val_loss:.6f} | "
              f"Train IoU: {t_iou:.4f} | Val IoU: {v_iou:.4f} | "
              f"Train MAE: {t_mae:.4f} | Val MAE: {v_mae:.4f} | "
              f"Train R2: {t_r2:.4f} | Val R2: {v_r2:.4f} | "
              f"Time: {epoch_time:.2f}s")

        train_log.append({'epoch': epoch, 'loss': train_loss, 'time': epoch_time, **train_metrics})
        val_log.append({'epoch': epoch, 'loss': val_loss, 'time': epoch_time, **val_metrics})

        # best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path_best)
            print(f"[INFO] Best model updated at epoch {epoch}, saved to {save_path_best}")

        # interval checkpoint
        if epoch % 50 == 0:
            ckpt_path = os.path.join(model_dir, f"epoch_{epoch:03d}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Checkpoint saved at {ckpt_path}")

    # 8. after training
    total_train_time = time.time() - train_start
    with open(his_path, 'a') as f:
        f.write(f"Total training time: {total_train_time:.2f} s\n")
    print(f"[INFO] Total training time: {total_train_time:.2f} s")

    pd.DataFrame(train_log).to_csv(log_train_csv, index=False)
    pd.DataFrame(val_log).to_csv(log_val_csv, index=False)
    print(f"[INFO] Training logs saved to {log_train_csv} and {log_val_csv}")


# if __name__ == "__main__":
#     gpu_id = 0
#     cfg = load_config('./config_example.yaml')
#     main(cfg, gpu_id=gpu_id)
