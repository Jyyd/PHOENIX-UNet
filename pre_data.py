# ====
# Arthor: JYYD 
# Date: 2025-11-07
# ====
import sys, os
os.chdir(sys.path[0])
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter, uniform_filter
from tqdm import tqdm
import random
random.seed(42)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'Arial'


# ============ Utility functions ============

def mean_smooth_no_building(data, building_mask, size):
    """
    Apply mean (uniform) smoothing on non-building regions only.
    Building pixels (mask == 1) remain zero.
    """
    smooth = uniform_filter(data, size=size, mode='constant', cval=0)
    smooth[building_mask == 1] = 0
    return smooth


def plot_sample(img, label, save_path, title=""):
    """
    Visualize 3-channel input (building, plume, source) and label.
    Save the figure as a .png file.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    ims = []
    ims.append(axes[0].imshow(img[0], cmap='gray', origin='lower'))
    axes[0].set_title("Building")
    ims.append(axes[1].imshow(img[1], cmap='inferno', origin='lower'))
    axes[1].set_title("Plume")
    ims.append(axes[2].imshow(img[2], cmap='Reds', origin='lower'))
    source_y, source_x = np.where(img[2] == 1)
    if len(source_x) > 0:
        axes[2].scatter(
            source_x, source_y,
            c='r', s=60, marker='o', linewidths=2, label='Source',
            alpha=0.1, edgecolors='r'
        )
    axes[2].set_title("Source (red o = 1)")
    axes[2].set_title("Source")
    building_mask = (img[0] > 0)
    label_masked = np.where(building_mask, np.nan, label)  # mask out buildings
    ims.append(axes[3].imshow(label_masked, cmap='Blues', origin='lower', vmin=0, vmax=10))
    axes[3].set_title("Label (log1p)")
    for ax in axes:
        ax.axis("off")
    for im, ax in zip(ims, axes):
        fig.colorbar(im, ax=ax, shrink=1, extend='max')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def preprocess_file(input_path, output_path, img_size=256):
    """
    Preprocess a single .npz file:
        - crop to 256×256 center region
        - smooth label (non-building region only)
        - apply log transform
        - normalize plume channel
        - binarize building mask
        - save processed arrays to .npz
    """
    try:
        data = np.load(input_path)

        # --- concatenate source + meteorology metadata ---
        source_vec = np.array(data["source_vec"]).reshape(-1).astype(np.float32)
        meteo_vec = np.array(data["meteo_vec"]).reshape(-1).astype(np.float32)
        meta = np.concatenate([source_vec, meteo_vec], axis=0)

        # --- crop label around center ---
        label = data["con_data_98"].astype(np.float32)
        middle_y, middle_x = 498, 538
        re_x_min, re_x_max = middle_x - img_size // 2, middle_x + img_size // 2
        re_y_min, re_y_max = middle_y - img_size // 2, middle_y + img_size // 2
        label = label[re_y_min:re_y_max, re_x_min:re_x_max]

        # --- crop 3-channel input image ---
        img = data["three_channel_data"].astype(np.float32)
        img = img[:, re_y_min:re_y_max, re_x_min:re_x_max]
        building_mask = (img[0] > 0).astype(np.uint8)

        # --- apply smoothing and log transforms ---
        label = mean_smooth_no_building(label, building_mask, size=3)
        label = gaussian_filter(label, sigma=2)
        label = np.log1p(label)

        # --- process Gaussian plume channel ---
        img[1] = np.log1p(img[1])
        minmax_scaler = MinMaxScaler()
        img[1] = minmax_scaler.fit_transform(img[1])

        # --- binarize building channel ---
        img[0] = (img[0] > 0).astype(img.dtype)

        # --- save preprocessed arrays ---
        np.savez_compressed(output_path, img=img, meta=meta, label=label)

        return img, label  # return for visualization
    except Exception as e:
        print(f"[ERROR] Failed to process {input_path}: {e}")
        return None, None


def preprocess_all(input_dir, txt_path, output_dir, fig_path=None, restart_flag=False):
    """
    Preprocess all .npz files listed in a text file.
    - If restart_flag=False: skip files that already exist.
    - If restart_flag=True: overwrite existing files.
    - After preprocessing, randomly select one newly generated file for visualization.
    - If no new files are generated, randomly select an existing file instead.
    - The representative figure is always overwritten.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(txt_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"\n[INFO] Start preprocessing {len(files)} files from {input_dir}")
    new_files = []  # store paths of newly generated npz files

    for fn in tqdm(files, desc=f"Processing {os.path.basename(txt_path)}"):
        input_path = os.path.join(input_dir, fn)
        out_name = f"pre_{os.path.basename(fn)}"
        output_path = os.path.join(output_dir, out_name)

        # Skip if already processed and restart_flag=False
        if os.path.exists(output_path) and not restart_flag:
            continue

        img, label = preprocess_file(input_path, output_path)
        if img is not None:
            new_files.append(output_path)

    # --- Always create or overwrite the representative figure ---
    if fig_path is not None:
        sample_path = None

        # Prefer a new file for visualization
        if new_files:
            sample_path = random.choice(new_files)
        else:
            # If no new files, randomly select an existing one
            all_files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
            if all_files:
                sample_path = os.path.join(output_dir, random.choice(all_files))

        if sample_path and os.path.exists(sample_path):
            data = np.load(sample_path)
            img, label = data["img"], data["label"]
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            plot_sample(img, label, fig_path, title=os.path.basename(fig_path).replace(".png", ""))
            print(f"[INFO] Saved representative figure to {fig_path}")
        else:
            print(f"[WARN] No available sample found to plot for {fig_path}")

    print(f"[INFO] Finished processing {len(files)} files. Results saved in {output_dir}\n")

# ============ Main entry ============

if __name__ == "__main__":
    # === paths (modify according to your folder structure) ===
    input_dir = "../Gas_apss/dataset/5min_m_apss_Data/"      # raw .npz directory
    output_dir = "./dataset/preprocessed_npz/" # where to save processed data
    fig_root = "./figures/"        # where to save visualization images

    # === define dataset splits ===
    splits = {
        "train": "./dataset/data_split/example_train.txt",
        "val": "./dataset/data_split/example_val.txt",
        "test": "./dataset/data_split/example_test.txt"
    }

    # === process all splits ===
    for split_name, txt_path in splits.items():
        out_split_dir = os.path.join(output_dir, split_name)
        fig_path = os.path.join(fig_root, f"{split_name}.png")  # e.g., pre_figures/train.png
        preprocess_all(input_dir, txt_path, out_split_dir, fig_path=fig_path)
