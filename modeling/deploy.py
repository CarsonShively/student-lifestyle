import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
import joblib
import json
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

y = (df["stress_level"].map(label_to_int).astype("int64").to_numpy())
X = df[feat_cols]

def X_to_float32(d: pd.DataFrame) -> pd.DataFrame:
    out = d.apply(pd.to_numeric, errors="coerce")
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

counts = np.bincount(y, minlength=3)
w_map  = counts.max() / np.clip(counts, 1, None)
w_all  = w_map[y]

pipe.fit(X, y, **{"lgbm__sample_weight": w_all})

joblib.dump(pipe, "student_lifestyle.pkl") 

with open("feature_columns_sl.json", "w") as f:
    json.dump(list(X.columns), f)