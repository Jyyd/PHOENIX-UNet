import sys, os
os.chdir(sys.path[0])
import yaml
import train_phen
import test_phen

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_config('./dataset/config_files/config_example.yaml')
    gpu_id = 0
    print(f"Trying on GPU {gpu_id} ...")
    train_phen.main(cfg, gpu_id)
    print("Training completed. Starting testing ...")
    test_phen.main(cfg, gpu_id)

