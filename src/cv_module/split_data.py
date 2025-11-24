# # src/cv_module/split_data.py
# import pandas as pd
# from pathlib import Path
# import random
# import shutil

# random.seed(42)

# META = Path("data/processed/metadata.csv")
# OUT_ROOT = Path("data/processed/images")
# OUT_ROOT.mkdir(parents=True, exist_ok=True)

# df = pd.read_csv(META)
# labels = df['label'].unique()

# # split ratios
# train_frac, val_frac, test_frac = 0.8, 0.1, 0.1

# for label in labels:
#     rows = df[df['label']==label].sample(frac=1, random_state=42).reset_index(drop=True)
#     n = len(rows)
#     n_train = int(n * train_frac)
#     n_val = int(n * val_frac)
#     train = rows.iloc[:n_train]
#     val = rows.iloc[n_train:n_train+n_val]
#     test = rows.iloc[n_train+n_val:]

#     for split_name, split_df in [("train",train),("val",val),("test",test)]:
#         for fp in split_df['filepath']:
#             src = Path(fp)
#             dest_dir = OUT_ROOT / split_name / label
#             dest_dir.mkdir(parents=True, exist_ok=True)
#             # Create symlink to save space (on Windows you may copy instead)
#             dest = dest_dir / src.name
#             try:
#                 dest.symlink_to(src.resolve())
#             except Exception:
#                 # fallback to copy if symlink not permitted
#                 shutil.copy2(src, dest)

# print("Done creating splits under", OUT_ROOT)


# src/cv_module/split_to_csv.py
import pandas as pd
from pathlib import Path
import random
import argparse
from collections import defaultdict

def stratified_split(df, label_col="label", train_frac=0.8, val_frac=0.1, test_frac=0.1, random_seed=42):
    random.seed(random_seed)
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    groups = df_shuffled.groupby(label_col)
    train_rows, val_rows, test_rows = [], [], []
    for _, grp in groups:
        n = len(grp)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_rows.append(grp.iloc[:n_train])
        val_rows.append(grp.iloc[n_train:n_train + n_val])
        test_rows.append(grp.iloc[n_train + n_val:])
    train_df = pd.concat(train_rows).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    val_df = pd.concat(val_rows).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    test_df = pd.concat(test_rows).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return train_df, val_df, test_df

def main(meta_path="data/processed/metadata.csv", out_dir="data/processed", train_frac=0.8, val_frac=0.1, test_frac=0.1):
    meta = Path(meta_path)
    if not meta.exists():
        raise FileNotFoundError(meta_path)
    df = pd.read_csv(meta)
    train_df, val_df, test_df = stratified_split(df, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "val.csv"
    test_csv = out_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Wrote: {train_csv} ({len(train_df)}), {val_csv} ({len(val_df)}), {test_csv} ({len(test_df)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="data/processed/metadata.csv")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()
    main(meta_path=args.meta, out_dir=args.out, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac)
