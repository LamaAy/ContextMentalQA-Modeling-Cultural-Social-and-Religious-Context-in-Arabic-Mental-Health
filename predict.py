import argparse, os, pandas as pd
from src.contextmentalqa.infer import predict_ensemble

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--text_col", type=str, default="question")
    ap.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    ap.add_argument("--thresholds", type=str, default=None)
    ap.add_argument("--output", type=str, default="outputs/inference.csv")
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--global_thr", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = predict_ensemble(
        df_infer=df,
        text_col=args.text_col,
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        max_len=args.max_len,
        thresholds_path=args.thresholds,
        global_threshold=args.global_thr
    )
    os.makedirs("outputs", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
