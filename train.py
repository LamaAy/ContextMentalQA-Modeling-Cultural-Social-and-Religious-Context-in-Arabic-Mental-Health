import argparse
from src.contextmentalqa.train import train_kfold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    train_kfold(args.config)

if __name__ == "__main__":
    main()
