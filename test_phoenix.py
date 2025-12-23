import os
import test
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Networks.Networks import PHOENIXUNet, CNNEmbed, OnlyUNet, OnlyCNN
from train_phoenix import get_loss_fn, compute_metrics, GasFieldDatasetPre
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
from tqdm import tqdm
import random
import yaml
import time

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def read_file_list(txt_path, npz_dir=None, add_prefix=False):
    with open(txt_path, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    if add_prefix:
        files = [f"pre_{fn}" if not fn.startswith("pre_") else fn for fn in files]
    if npz_dir is not None:
        files = [os.path.join(npz_dir, fn) for fn in files]
    return files
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def plot_loss_curves(train_log_csv, val_log_csv, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    train_log = pd.read_csv(train_log_csv)
    val_log = pd.read_csv(val_log_csv)
    metrics = ['loss', 'mae', 'rmse', 'iou', 'r2']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(train_log['epoch'], train_log[metric], label=f"Train {metric}")
        plt.plot(val_log['epoch'], val_log[metric], label=f"Val {metric}")
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.title(f"{metric} Curve")
        plt.savefig(os.path.join(save_dir, f'{metric}_curve.png'), dpi=300)
        plt.tight_layout
        plt.close()
    print(f"Loss and metrics curves saved in {save_dir}")

def test_and_save_pred(model, test_loader, device, out_dir, fig_out_dir=None):
    """
    Evaluate model on test set and save results.
    - Computes per-sample metrics (MAE, MSE, RMSE, IoU, R2)
    - Saves all predictions to .npz and metrics to .csv
    - Optionally saves one visualization figure (first sample)
    """
    model.eval()
    preds, trues = [], []
    all_metrics = []
    os.makedirs(out_dir, exist_ok=True)
    if fig_out_dir:
        os.makedirs(fig_out_dir, exist_ok=True)

    start_time = time.time()

    for i, batch in enumerate(tqdm(test_loader, desc="[Testing]", ncols=100)):
        if batch is None:
            continue
        img, meta, label = batch
        img, meta, label = img.to(device), meta.to(device), label.to(device)

        # --- forward ---
        with torch.no_grad():
            pred = model(img, meta)

        preds.append(pred.detach().cpu().numpy())
        trues.append(label.detach().cpu().numpy())

        # --- compute per-sample metrics ---
        mask = (label > 0).float()
        metrics = compute_metrics(pred, label, mask=mask)
        all_metrics.append(metrics)
    # --- summary ---
    total_time = time.time() - start_time
    num_samples = len(test_loader.dataset)
    avg_time = total_time / max(num_samples, 1)

    print(f"\n[Test Summary]")
    print(f"Total inference time: {total_time:.2f} s")
    print(f"Average time per sample: {avg_time:.4f} s")

    # --- save all predictions ---
    if preds:
        np.savez_compressed(
            os.path.join(out_dir, 'all_test_pred.npz'),
            preds=np.concatenate(preds, axis=0),
            trues=np.concatenate(trues, axis=0)
        )
        print(f"Saved all predictions to {out_dir}")

    # --- metrics ---
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_csv = os.path.join(out_dir, 'test_metrics.csv')
        metrics_df.to_csv(metrics_csv, index=False)
        mean_metrics = metrics_df.mean().to_dict()

        print("\n[Test Metrics] Average across all samples:")
        for k, v in mean_metrics.items():
            print(f"{k:<10s}: {v:>10.6f}")
    else:
        mean_metrics = {}

    return mean_metrics

# -----------------------------
# main
# -----------------------------
def main(cfg, gpu_id=None):
    # 1. basic settings
    set_seed(cfg['seed'])
    npz_train_dir = cfg['npz_train_dir']
    npz_val_dir = cfg['npz_val_dir']
    npz_test_dir = cfg['npz_test_dir']
    train_txt = cfg['train_txt']
    val_txt = cfg['val_txt']
    test_txt = cfg['test_txt']
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    exp_name = f"FHENUNet_bs{batch_size}_lr{lr}_loss{loss_type}_ch{base_ch}_dim{fuse_dim}_depth{depth}_{model_type}"
    save_dir = f'./results/exps/{exp_name}'
    model_dir = os.path.join(save_dir, 'model')
    log_dir = os.path.join(save_dir, 'log')
    pred_dir = os.path.join(save_dir, 'pred_npz')
    test_pred_dir = os.path.join(pred_dir, 'test_pred_npz')
    fig_dir = os.path.join(save_dir, 'figures')
    test_figure_dir = os.path.join(fig_dir, 'test_figures')
    fig_dir = os.path.join(save_dir, 'figures')
    for d in [model_dir, log_dir, pred_dir, test_pred_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    weight_path = f'{model_dir}/best_model.pt'
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"Loaded weights from {weight_path}")

    test_files = read_file_list(test_txt, npz_test_dir, add_prefix=True)
    test_dataset = GasFieldDatasetPre(test_files)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    print(f"Loaded {len(test_loader)} test samples from {test_txt}")

    test_and_save_pred(model, test_loader, device, out_dir=test_pred_dir,
                       fig_out_dir=test_figure_dir)