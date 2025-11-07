# ====
# Arthor: JYYD 
# Date: 2025-11-07
# ====
import sys, os
os.chdir(sys.path[0])
import numpy as np
import pandas as pd
import re
import geopandas as gpd
import seaborn as sns
from sklearn.model_selection import train_test_split

def parse_filename(filename):
    '''
    Arguments:
        filename: str, e.g. min5_m_apss_v5_5_d90_sc4_s00001.npz
    '''
    # regex pattern to match the filename format
    pattern = re.compile(
        r"min5_m_apss_v([0-9]+)_([0-9]+)_d([-0-9.]+)_sc([0-9]+)_s([0-9]+)_([0-9]{5})\.npz"
    )
    m = pattern.match(filename)
    if m:
        wind_speed = float(f"{m.group(1)}.{m.group(2)}")  # v5_5 -> 5.5
        wind_direction = int(float(m.group(3)))
        stability = int(m.group(4))
        source_num = int(m.group(5))
        return wind_speed, wind_direction, stability, source_num
    else:
        raise ValueError(f"Cannot parse {filename}")

def random_split_sources(
    num_sources=98,           # source points total
    select_num=60,            # number of points to select
    train_ratio=0.8, 
    testval_ratio=0.5, 
    seed=42
):
    """
    select select_num from num_sources, then split into train/val/test, return sorted lists
    """
    all_src_nums = np.arange(1, num_sources+1)
    np.random.seed(seed)
    selected_srcs = np.random.choice(all_src_nums, size=select_num, replace=False)  # 先选出60个

    np.random.shuffle(selected_srcs)
    n_train = int(train_ratio * select_num)
    n_val_test = select_num - n_train
    n_val = int(testval_ratio * n_val_test)
    n_test = n_val_test - n_val

    train_src = sorted(selected_srcs[:n_train])
    val_src = sorted(selected_srcs[n_train:n_train+n_val])
    test_src = sorted(selected_srcs[n_train+n_val:])
    return train_src, val_src, test_src


def dataset_split(
    npz_dir,
    fields_design=None,
    save_dir=None
):
    # 1. gather all files & extract fields
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    info = []
    for fname in npz_files:
        try:
            ws, wd, sc, sn = parse_filename(fname)
            info.append({'fname': fname, 'wind_direction': wd, 'source_num': sn,
                         'wind_speed': ws, 'stability': sc})
        except Exception as e:
            print('Warning:', e)
    df = pd.DataFrame(info)

    wind_mask = df['wind_direction'].isin(fields_design.get('wind_direction', []))
    train_src_mask = df['source_num'].isin(fields_design.get('train_source_num', []))
    val_src_mask = df['source_num'].isin(fields_design.get('val_source_num', []))
    test_src_mask = df['source_num'].isin(fields_design.get('test_source_num', []))

    train_mask = (wind_mask) & (train_src_mask)
    val_mask = (wind_mask) & (val_src_mask)
    test_mask = (wind_mask) & (test_src_mask)
    train_files = df[train_mask]['fname'].tolist()
    val_files = df[val_mask]['fname'].tolist()
    test_files = df[test_mask]['fname'].tolist()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, files in zip(['train','val','test'], [train_files, val_files, test_files]):
            save_path = os.path.join(save_dir, f"example_{name}.txt")
            with open(save_path, 'w') as f:
                for x in files:
                    f.write(x + '\n')
        print(f"Saved split file lists to: {save_dir}")
    print(f"Split summary: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return train_files, val_files, test_files, df

def extract_source_nums_from_files(filelist):
    return np.unique([int(re.search(r'_s([0-9]+)_', f).group(1)) for f in filelist])


# ================== Main ==================
if __name__ == "__main__":
    npz_dir = '../Gas_apss/dataset/5min_m_apss_Data/'
    # npz_dir = './dataset//5min_m_apss_Data/' # download dataset in this path
    source_csv = './dataset/ori_data/shp_point/source.csv'
    shapefile_path = './dataset/ori_data/shp_point/m_700.shp'
    save_dir = './dataset/data_split'
    seed = 42
    num_sources = 98
    select_num = 60

    # randomly select source points and split into train/val/test
    train_src, val_src, test_src = random_split_sources(num_sources, select_num,
                                                        train_ratio=0.85,
                                                        testval_ratio=0.5, seed=seed)
    # print(train_src, val_src, test_src)
    def filter_files_by_src(files, src_nums):
        return [f for f in files if int(re.search(r'_s([0-9]+)_', f).group(1)) in src_nums]
    all_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    train_files = filter_files_by_src(all_files, train_src)
    val_files = filter_files_by_src(all_files, val_src)
    test_files = filter_files_by_src(all_files, test_src)

    fields_design = {'wind_direction': [0, 90, 180, 270],
                     'train_source_num': list(train_src),
                     'val_source_num': list(val_src),
                     'test_source_num': list(test_src)}


    train_files, val_files, test_files, df = dataset_split(
        npz_dir,
        fields_design=fields_design,
        save_dir=save_dir
    )

    train_files = [x.strip() for x in open(f'{save_dir}/example_train.txt')]
    val_files = [x.strip() for x in open(f'{save_dir}/example_val.txt')]
    test_files = [x.strip() for x in open(f'{save_dir}/example_test.txt')]

    train_src = set(map(int, extract_source_nums_from_files(train_files)))
    val_src = set(map(int, extract_source_nums_from_files(val_files)))
    test_src = set(map(int, extract_source_nums_from_files(test_files)))

    print('\n=== Source Number Check ===')
    print('Checking source number overlaps:')
    print('val ∩ test:', [int(x) for x in sorted(val_src & test_src)])
    print('val ∩ train:', [int(x) for x in sorted(val_src & train_src)])
    print('test ∩ train:', [int(x) for x in sorted(test_src & train_src)])
    print('-----------------------------------------------')
    print('All good if above are empty sets.')
    print('train: ', len(train_src), 'val: ', len(val_src), 'test: ', len(test_src))
    print('train sources:', sorted(train_src),
          'val sources:', sorted(val_src), 'test sources:', sorted(test_src))
    print('===============================================')
