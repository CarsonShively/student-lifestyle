import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from huggingface_hub import hf_hub_download

train_data = hf_hub_download(
        repo_id="Carson-Shively/uber-fares",
        filename="data/gold/gold_uf.parquet",
        repo_type="dataset",
        revision="main",
    )

df = pd.read_parquet(train_data)

label_to_int = {'Low': 0, 'Moderate': 1, 'High': 2}

feat_cols = [
    "study","extra","sleep","social","pa",
    "total_active","overscheduled",
    "study_share","social_share","extra_share","pa_share",
    "study_to_sleep","study_to_social",
    "sleep_deficit","sleep_abs_delta","sleep_low","sleep_high",
    "pa_guideline","pa_under",
    "study_c2","sleep_c2","social_c2",
    "sleep_x_study","study_x_social",
    "gpa"
]

monotone = [
    0,0,0,0,0,
    0,+1,
    0,0,0,0,
    0,0,
    +1,0,0,0,
    -1,+1,
    0,0,0,
    0,0,
    0
]
assert len(feat_cols) == len(monotone) == 25

y = df["stress_level"].map(label_to_int).astype(int).to_numpy()
X = df[feat_cols]

def X_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(np.float32)

pre = FunctionTransformer(X_to_float32, validate=False)

lgbm = LGBMRegressor(
    objective="regression",
    monotone_constraints=monotone,
    n_estimators=800, learning_rate=0.05,
    num_leaves=15, max_depth=5,
    min_child_samples=40, min_child_weight=5.0,
    min_split_gain=0.0, reg_alpha=0.0, reg_lambda=2.0,
    feature_fraction=0.85, bagging_fraction=0.7, bagging_freq=1,
    n_jobs=-1, random_state=42, verbosity=-1,
)

pipe = Pipeline([
    ("pre", pre),
    ("lgbm", lgbm),
]).set_output(transform="pandas")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


oof_pred = np.zeros_like(y, dtype=float)
fold_val_idx = []

qwk_list, f1_list, bal_list, mae_list = [], [], [], []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X.values, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    counts = np.bincount(y_tr, minlength=3)
    w_map  = counts.max() / np.clip(counts, 1, None)
    w_tr   = w_map[y_tr]

    X_val_t = X_to_float32(X_val)

    pipe.fit(
        X_tr, y_tr,
        **{
            "lgbm__sample_weight": w_tr,
            "lgbm__eval_set": [(X_val_t, y_val)],
            "lgbm__eval_metric": "l2",
            "lgbm__callbacks": [early_stopping(200, verbose=False), log_evaluation(0)],
        }
    )

    best_it = pipe.named_steps["lgbm"].best_iteration_
    yhat = pipe.named_steps["lgbm"].predict(X_val_t, num_iteration=best_it)

    oof_pred[val_idx] = yhat
    fold_val_idx.append(val_idx)

    y_idx = np.clip(np.rint(yhat), 0, 2).astype(int)
    qwk = cohen_kappa_score(y_val, y_idx, weights='quadratic')
    f1m = f1_score(y_val, y_idx, average='macro')
    bal = balanced_accuracy_score(y_val, y_idx)
    mae = np.mean(np.abs(y_val - y_idx))

    print(f"Fold {fold}: QWK={qwk:.3f} | Macro-F1={f1m:.3f} | BalAcc={bal:.3f} | MAE_ord={mae:.3f}")
    qwk_list.append(qwk); f1_list.append(f1m); bal_list.append(bal); mae_list.append(mae)

qwk_arr, f1_arr = np.array(qwk_list), np.array(f1_list)
bal_arr, mae_arr = np.array(bal_list), np.array(mae_list)

print("\nAverages (mean ± std):")
print(f"QWK      = {qwk_arr.mean():.3f} ± {qwk_arr.std():.3f}")
print(f"Macro-F1 = {f1_arr .mean():.3f} ± {f1_arr .std():.3f}")
print(f"BalAcc   = {bal_arr.mean():.3f} ± {bal_arr.std():.3f}")
print(f"MAE_ord  = {mae_arr.mean():.3f} ± {mae_arr.std():.3f}")

y_oof_idx = np.clip(np.rint(oof_pred), 0, 2).astype(int)
print("\nOverall OOF:")
print(f"QWK={cohen_kappa_score(y, y_oof_idx, weights='quadratic'):.3f} | "
      f"Macro-F1={f1_score(y, y_oof_idx, average='macro'):.3f} | "
      f"BalAcc={balanced_accuracy_score(y, y_oof_idx):.3f} | "
      f"MAE_ord={np.mean(np.abs(y - y_oof_idx)):.3f}")
print("\nConfusion matrix [Low, Moderate, High]:")
print(confusion_matrix(y, y_oof_idx, labels=[0,1,2]))

