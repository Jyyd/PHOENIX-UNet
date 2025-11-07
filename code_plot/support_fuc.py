import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
os.chdir(sys.path[0])
import matplotlib
import random

def get_metrics_data():
    exp2_metrics = pd.read_csv('../results/exp2_metrics2.csv')
    exp2_metrics = exp2_metrics.iloc[:, 1:]
    split_df = exp2_metrics['exp_name'].str.split('_', expand=True)
    split_df.columns = ['model', 'bs', 'lr', 'img', 'k', 'loss', 'ch', 'dim',
                        'depth', 'detail']
    split_df['bs'] = split_df['bs'].str.replace('bs', '').astype(int)
    split_df['lr'] = split_df['lr'].str.replace('lr', '').astype(float)
    split_df['img'] = split_df['img'].str.replace('img', '').astype(int)
    split_df['ch']  = split_df['ch'].str.replace('ch', '').astype(int)
    split_df['dim'] = split_df['dim'].str.replace('dim', '').astype(int)
    split_df['depth'] = split_df['depth'].str.replace('depth', '').astype(int)
    split_df['detail'] = split_df['detail'].str.replace('detail', '').astype(str)
    split_df = split_df[['bs', 'lr', 'img', 'ch', 'dim', 'depth', 'detail']]
    exp2_metrics_final = pd.concat([exp2_metrics, split_df], axis=1)
    return exp2_metrics_final

def get_train_metrics_data():
    exp2_metrics = pd.read_csv('../results/exp2_train_metrics.csv')
    split_df = exp2_metrics['exp_name'].str.split('_', expand=True)
    split_df.columns = ['model', 'bs', 'lr', 'img', 'k', 'loss', 'ch', 'dim',
                        'depth', 'detail']
    split_df['bs'] = split_df['bs'].str.replace('bs', '').astype(int)
    split_df['lr'] = split_df['lr'].str.replace('lr', '').astype(float)
    split_df['img'] = split_df['img'].str.replace('img', '').astype(int)
    split_df['ch']  = split_df['ch'].str.replace('ch', '').astype(int)
    split_df['dim'] = split_df['dim'].str.replace('dim', '').astype(int)
    split_df['depth'] = split_df['depth'].str.replace('depth', '').astype(int)
    split_df['detail'] = split_df['detail'].str.replace('detail', '').astype(str)
    split_df = split_df[['bs', 'lr', 'img', 'ch', 'dim', 'depth', 'detail']]
    exp2_metrics_final = pd.concat([exp2_metrics, split_df], axis=1)
    return exp2_metrics_final

def get_val_metrics_data():
    exp2_metrics = pd.read_csv('../results/exp2_val_metrics.csv')
    split_df = exp2_metrics['exp_name'].str.split('_', expand=True)
    split_df.columns = ['model', 'bs', 'lr', 'img', 'k', 'loss', 'ch', 'dim',
                        'depth', 'detail']
    split_df['bs'] = split_df['bs'].str.replace('bs', '').astype(int)
    split_df['lr'] = split_df['lr'].str.replace('lr', '').astype(float)
    split_df['img'] = split_df['img'].str.replace('img', '').astype(int)
    split_df['ch']  = split_df['ch'].str.replace('ch', '').astype(int)
    split_df['dim'] = split_df['dim'].str.replace('dim', '').astype(int)
    split_df['depth'] = split_df['depth'].str.replace('depth', '').astype(int)
    split_df['detail'] = split_df['detail'].str.replace('detail', '').astype(str)
    split_df = split_df[['bs', 'lr', 'img', 'ch', 'dim', 'depth', 'detail']]
    exp2_metrics_final = pd.concat([exp2_metrics, split_df], axis=1)
    return exp2_metrics_final

def get_log_data():
    exp2_log_data = pd.read_csv('../results/exp2_logs_all2.csv')
    exp2_log_data = exp2_log_data.iloc[:, 1:]
    split_log = exp2_log_data['exp_name'].str.split('_', expand=True)
    split_log.columns = ['model', 'bs', 'lr', 'img', 'k', 'loss', 'ch', 'dim',
                         'depth', 'detail']
    split_log['bs'] = split_log['bs'].str.replace('bs', '').astype(int)
    split_log['lr'] = split_log['lr'].str.replace('lr', '').astype(float)
    split_log['img'] = split_log['img'].str.replace('img', '').astype(int)
    split_log['ch']  = split_log['ch'].str.replace('ch', '').astype(int)
    split_log['dim'] = split_log['dim'].str.replace('dim', '').astype(int)
    split_log['depth'] = split_log['depth'].str.replace('depth', '').astype(int)
    split_log['detail'] = split_log['detail'].str.replace('detail', '').astype(str)
    split_log = split_log[['bs', 'lr', 'img', 'ch', 'dim', 'depth', 'detail']]
    exp2_log_data_final = pd.concat([exp2_log_data, split_log], axis=1)
    exp2_log_data_final_val = exp2_log_data_final[exp2_log_data_final['type'] == 'val']
    exp2_log_data_final_train = exp2_log_data_final[exp2_log_data_final['type'] == 'train']
    return exp2_log_data_final_val, exp2_log_data_final_train

def get_epoch_data(data, bs_size):
    base_data = data[(data['bs'] == bs_size) &
                   (data['lr'] == 0.001) &
                    (data['img'] == 256) &
                    (data['ch'] == 32) &
                    (data['dim'] == 16) &
                    (data['depth'] == 4) &
                    (data['detail'] == 'None')].reset_index(drop=True)
    base_data['epochs_num'] = 100
    data = data[(data['bs'] == bs_size) &
                   (data['lr'] == 0.001) &
                    (data['img'] == 256) &
                    (data['ch'] == 32) &
                    (data['dim'] == 16) &
                    (data['depth'] == 4) & 
                 (data['detail'].str.contains('epoch'))].reset_index(drop=True)
    data['epochs_num'] = data['detail'].str.replace('nepochs', '').astype(int)
    data_all = pd.concat([base_data, data], axis=0).reset_index(drop=True)
    data_all = data_all.sort_values(by='epochs_num').reset_index(drop=True)
    return data_all

def get_epoch_data_seed(data, bs_size):
    base_data = data[(data['bs'] == bs_size) &
                   (data['lr'] == 0.001) &
                    (data['img'] == 256) &
                    (data['ch'] == 32) &
                    (data['dim'] == 16) &
                    (data['depth'] == 4) &
                    (data['detail'] == 'None')].reset_index(drop=True)
    base_data['epochs_num'] = 100
    base_data['seed_num'] = 42
    data = data[(data['bs'] == bs_size) &
                   (data['lr'] == 0.001) &
                    (data['img'] == 256) &
                    (data['ch'] == 32) &
                    (data['dim'] == 16) &
                    (data['depth'] == 4) & 
                 (data['detail'].str.contains('sedd'))].reset_index(drop=True)
    data['seed_num'] = data['detail'].str.replace('sedd', '').astype(int)
    data['epochs_num'] = 100
    data_all = pd.concat([base_data, data], axis=0).reset_index(drop=True)
    data_all = data_all.sort_values(by='seed_num').reset_index(drop=True)
    return data_all

def get_epoch_num_data(data, epoch_num):
    epoch_data = data[data['epochs_num'] == epoch_num].reset_index(drop=True)
    epoch_data = epoch_data.sort_values(by='epoch').reset_index(drop=True)
    return epoch_data

def plot_hist_param_architecture(plot_data, plot_param, y_label,
                                 ylimits_l, ylimits_r, save_flag=False, save_folder=''):
    x_labels = plot_data[plot_param].astype(str)
    x_pos = np.arange(len(x_labels))
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax1.bar(x_pos, plot_data['r2'], 
                alpha=0.4, color="#c8dfef",
                label=r'${R^2}$', edgecolor='#1f77b4', linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}",
                ha='center', va='bottom', color='black', zorder=100)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel(r'${R^2}$')
    ax1.set_ylim(ylimits_l)
    ax1.set_xlabel(y_label)

    # twin y-axis 折线图也要改成用 x_pos 对应
    ax2 = ax1.twinx()
    ax2.plot(x_pos, plot_data['mse'],
            linestyle='-', color="#7cbbe8", markeredgecolor='k',
            marker='o', linewidth=1.5, alpha=0.8, label='MSE', markersize=10)
    ax2.plot(x_pos, plot_data['rmse'],
            linestyle='--', color="#3c749b", markeredgecolor='k',
            marker='p', linewidth=1.5, alpha=0.8, label='RMSE', markersize=11)
    ax2.plot(x_pos, plot_data['mae'],
            linestyle='-.', color="#0a283e", markeredgecolor='k',
            marker='s', linewidth=1.5, alpha=0.8, label='MAE', markersize=10)

    ax2.set_ylabel('Metric Values')
    ax2.set_ylim(ylimits_r)

    # legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2,
            labels + labels2,
            loc='upper center', ncol=4,
            bbox_to_anchor=(0.5, 1.2),
            edgecolor='black')

    plt.tight_layout()
    if save_flag:
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.png', dpi=300)
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.svg', dpi=300, format='svg')
    plt.show()


def plot_hist_param_input(plot_data, plot_param, y_label,
                                 ylimits_l, ylimits_r, save_flag=False,
                                 save_folder=''):
    x_labels = plot_data[plot_param].astype(str)
    x_pos = np.arange(len(x_labels))
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax1.bar(x_pos, plot_data['r2'], 
                alpha=0.4, color="#efc8c8",
                label=r'${R^2}$', edgecolor="#b41f1f", linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        if height < 0.1:
            continue
        else:
             ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}",
                        ha='center', va='bottom', color='black', zorder=100)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel(r'${R^2}$')
    ax1.set_ylim(ylimits_l)
    ax1.set_xlabel(y_label)

    # twin y-axis 折线图也要改成用 x_pos 对应
    ax2 = ax1.twinx()
    ax2.plot(x_pos, plot_data['mse'],
            linestyle='-', color="#e87c7c", markeredgecolor='k',
            marker='o', linewidth=1.5, alpha=0.8, label='MSE', markersize=10)
    ax2.plot(x_pos, plot_data['rmse'],
            linestyle='--', color="#9b3c3c", markeredgecolor='k',
            marker='p', linewidth=1.5, alpha=0.8, label='RMSE', markersize=11)
    ax2.plot(x_pos, plot_data['mae'],
            linestyle='-.', color="#3e0a0a", markeredgecolor='k',
            marker='s', linewidth=1.5, alpha=0.8, label='MAE', markersize=10)

    ax2.set_ylabel('Metric Values')
    ax2.set_ylim(ylimits_r)

    # legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2,
            labels + labels2,
            loc='upper center', ncol=4,
            bbox_to_anchor=(0.5, 1.2),
            edgecolor='black')

    plt.tight_layout()
    if save_flag:
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.png', dpi=300)
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.svg', dpi=300, format='svg')
    plt.show()


def plot_hist_param_train(plot_data, plot_param, y_label,
                          ylimits_l, ylimits_r, save_flag=False,
                          save_folder=''):
    x_labels = plot_data[plot_param].astype(str)
    x_pos = np.arange(len(x_labels))
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax1.bar(x_pos, plot_data['r2'], 
                alpha=0.4, color="#cbefc8",
                label=r'${R^2}$', edgecolor="#26b41f", linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        if height < 0.1:
            continue
        else:
             ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}",
                        ha='center', va='bottom', color='black', zorder=100)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel(r'${R^2}$')
    ax1.set_ylim(ylimits_l)
    ax1.set_xlabel(y_label)

    # twin y-axis 折线图也要改成用 x_pos 对应
    ax2 = ax1.twinx()
    ax2.plot(x_pos, plot_data['mse'],
            linestyle='-', color="#85e87c", markeredgecolor='k',
            marker='o', linewidth=1.5, alpha=0.8, label='MSE', markersize=10)
    ax2.plot(x_pos, plot_data['rmse'],
            linestyle='--', color="#479b3c", markeredgecolor='k',
            marker='p', linewidth=1.5, alpha=0.8, label='RMSE', markersize=11)
    ax2.plot(x_pos, plot_data['mae'],
            linestyle='-.', color="#0a3e10", markeredgecolor='k',
            marker='s', linewidth=1.5, alpha=0.8, label='MAE', markersize=10)

    ax2.set_ylabel('Metric Values')
    ax2.set_ylim(ylimits_r)

    # legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2,
            labels + labels2,
            loc='upper center', ncol=4,
            bbox_to_anchor=(0.5, 1.2),
            edgecolor='black')

    plt.tight_layout()
    if save_flag:
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.png', dpi=300)
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.svg', dpi=300, format='svg')
    plt.show()

def plot_hist_param_train_special(plot_data, plot_param, y_label,
                          ylimits_l, ylimits_r, save_flag=False,
                          save_folder=''):
    x_labels = plot_data[plot_param].astype(str)
    x_pos = np.arange(len(x_labels))
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax1.bar(x_pos, plot_data['r2'], 
                alpha=0.4, color="#cbefc8",
                label=r'${R^2}$', edgecolor="#26b41f", linewidth=1.5)
    for bar in bars:
        height = bar.get_height()
        if height < 0.1:
            continue
        else:
             ax1.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}",
                        ha='center', va='bottom', color='black', zorder=100)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel(r'${R^2}$')
    ax1.set_ylim(ylimits_l)
    ax1.set_xlabel(y_label)

    # twin y-axis 折线图也要改成用 x_pos 对应
    ax2 = ax1.twinx()
    ax2.plot(x_pos, plot_data['mse'],
            linestyle='-', color="#85e87c", markeredgecolor='k',
            marker='o', linewidth=1.5, alpha=0.8, label='MSE', markersize=10)
    ax2.plot(x_pos, plot_data['rmse'],
            linestyle='--', color="#479b3c", markeredgecolor='k',
            marker='p', linewidth=1.5, alpha=0.8, label='RMSE', markersize=11)
    ax2.plot(x_pos, plot_data['mae'],
            linestyle='-.', color="#0a3e10", markeredgecolor='k',
            marker='s', linewidth=1.5, alpha=0.8, label='MAE', markersize=10)

    ax2.set_ylabel('Metric Values')
    ax2.set_ylim(ylimits_r)

    # legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2,
            labels + labels2,
            loc='upper center', ncol=4,
            bbox_to_anchor=(0.5, 1.2),
            edgecolor='black')

    plt.tight_layout()
    if save_flag:
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.png', dpi=300)
        plt.savefig(f'../fig_figures/hyperparameter/{save_folder}/hist_{plot_param}_param.svg', dpi=300, format='svg')
    plt.show()

def pad_center_crop(arr, center_y, center_x, out_h=256, out_w=256):
    if arr.ndim == 3:
        C, H, W = arr.shape
        out = np.zeros((C, out_h, out_w), dtype=arr.dtype)
    else:
        H, W = arr.shape
        out = np.zeros((out_h, out_w), dtype=arr.dtype)

    y0, x0 = center_y - out_h // 2, center_x - out_w // 2
    y1, x1 = y0 + out_h, x0 + out_w

    sy0, sy1 = max(0, y0), min(H, y1)
    sx0, sx1 = max(0, x0), min(W, x1)

    dy0, dx0 = sy0 - y0, sx0 - x0
    dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)

    if arr.ndim == 3:
        out[:, dy0:dy1, dx0:dx1] = arr[:, sy0:sy1, sx0:sx1]
    else:
        out[dy0:dy1, dx0:dx1] = arr[sy0:sy1, sx0:sx1]
    return out

def get_building_area():
    # load building data
    npz_path = '../dataset_m/5min_m_Data/min5_m_v0_5_d0_sc2_s19_00011.npz'
    # npz_path = 'F://Program//CODE//GAS//Gas_unet//Gas_code//dataset_m//min5_m_v0_5_d0_sc2_s19_00011.npz'
    data = np.load(npz_path)
    build_data = data['three_channel_data'][0]
    non_building_mask = (build_data == 0).astype(np.uint8)
    center_y, center_x = 498, 538
    build_data_256 = pad_center_crop(build_data, center_y, center_x, 256, 256)
    non_building_mask = pad_center_crop(non_building_mask, center_y, center_x, 256, 256)
    return build_data_256, non_building_mask

def get_indices(test_metrics):
    r2_result = pd.DataFrame(columns=['r2_90', 'r2_80_90', 'r2_60_80',
                                      'r2_0_60', 'r2_neg'])
    r2_90 = test_metrics[test_metrics['r2'] >= 0.9].shape[0]
    r2_80_90 = test_metrics[(test_metrics['r2'] >= 0.8) & (test_metrics['r2'] < 0.9)].shape[0]
    r2_60_80 = test_metrics[(test_metrics['r2'] >= 0.6) & (test_metrics['r2'] < 0.8)].shape[0]
    r2_40_60 = test_metrics[(test_metrics['r2'] >= 0.4) & (test_metrics['r2'] < 0.6)].shape[0]
    r2_20_40 = test_metrics[(test_metrics['r2'] >= 0.2) & (test_metrics['r2'] < 0.4)].shape[0]
    r2_0_20 = test_metrics[(test_metrics['r2'] >= 0.0) & (test_metrics['r2'] < 0.2)].shape[0]
    r2_neg = test_metrics[test_metrics['r2'] < 0.0].shape[0]
    r2_result = r2_result._append({
        'r2_90': np.round(r2_90/test_metrics.shape[0]*100, 4),
        'r2_80_90': np.round(r2_80_90/test_metrics.shape[0]*100, 4),
        'r2_60_80': np.round(r2_60_80/test_metrics.shape[0]*100, 4),
        'r2_40_60': np.round(r2_40_60/test_metrics.shape[0]*100, 4),
        'r2_20_40': np.round(r2_20_40/test_metrics.shape[0]*100, 4),
        'r2_0_20': np.round(r2_0_20/test_metrics.shape[0]*100, 4),
        'r2_neg': np.round(r2_neg/test_metrics.shape[0]*100, 4)
    }, ignore_index=True)
    
    # 找到最好的五个r2
    best_indices = test_metrics['r2'].nlargest(5).index.tolist()
    worest_indices = test_metrics['r2'].nsmallest(5).index.tolist()
    plot_indices = []
    r2_90_indices = test_metrics[test_metrics['r2'] >= 0.9].index.tolist()
    r2_80_90_indices = test_metrics[(test_metrics['r2'] >= 0.8) & (test_metrics['r2'] < 0.9)].index.tolist()
    r2_60_80_indices = test_metrics[(test_metrics['r2'] >= 0.6) & (test_metrics['r2'] < 0.8)].index.tolist()
    r2_0_60_indices = test_metrics[(test_metrics['r2'] >= 0.0) & (test_metrics['r2'] < 0.6)].index.tolist()
    r2_80_indices = test_metrics[(test_metrics['r2']>= 0.8)].index.tolist()
    r2_60_indices = test_metrics[(test_metrics['r2']>= 0.6)].index.tolist()
    # if len(r2_0_60_indices) > 0:
    #     plot_indices.append(random.choice(r2_0_60_indices))
    # if len(r2_90_indices) > 2:
    #     plot_indices.extend(random.sample(r2_90_indices, 2))
    # if len(r2_80_90_indices) > 1:
    #     plot_indices.extend(random.sample(r2_80_90_indices, 1))
    # if len(r2_60_80_indices) > 1:
    #     plot_indices.extend(random.sample(r2_60_80_indices, 1))
    if len(r2_80_indices) >= 2:
        plot_indices.extend(random.sample(r2_80_indices, 2))
    else:
        plot_indices.extend(random.sample(r2_60_indices, 2))
    return plot_indices, best_indices, worest_indices, r2_result