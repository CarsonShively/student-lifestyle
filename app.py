import sys
import types
import json
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import joblib
import gradio as gr
from pydantic import BaseModel, ConfigDict, ValidationError
import threading, types, duckdb
from importlib.resources import files
import os

state = types.SimpleNamespace()
_duck_lock = threading.Lock()

def X_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.where(np.isfinite(out), np.nan)
    return out.astype(np.float32)

m = sys.modules.get('__main__') or types.ModuleType('__main__')
sys.modules['__main__'] = m
for name, fn in {
    'X_to_float32': X_to_float32,
}.items():
    setattr(m, name, fn)

REPO_ID = "Carson-Shively/student-lifestyle"
REV = "main" 

ROOT = Path(__file__).resolve().parent

MODEL_PKL_PATH  = ROOT / "artifacts" / "student_lifestyle.pkl"
FEATS_JSON_PATH = ROOT / "artifacts" / "feature_columns.json"
SQL_PKG = "student_lifestyle.data_layers.gold"
MACROS_SQL_FILE = "macros.sql"
ONLINE_SQL_FILE = "online.sql"

def _read_pkg_sql(pkg: str, filename: str) -> str:
    return (files(pkg) / filename).read_text(encoding="utf-8")

def init_connection(con) -> None:
    con.execute(_read_pkg_sql(SQL_PKG, MACROS_SQL_FILE))


def load_model_and_schema():
    try:
        model = joblib.load(MODEL_PKL_PATH)
        feature_columns = json.loads(FEATS_JSON_PATH.read_text(encoding="utf-8"))
        return model, feature_columns
    except Exception as e:
        raise RuntimeError("Failed model/schema load") from e


def init_app():
    state.con = duckdb.connect()
    init_connection(state.con)
    state.MODEL, state.FEATURE_COLUMNS = load_model_and_schema()
    state.FEATURE_COLUMNS = tuple(state.FEATURE_COLUMNS)
    with _duck_lock:
        state.con.execute(_read_pkg_sql(SQL_PKG, ONLINE_SQL_FILE))

def collect_raw_inputs(study, extra, sleep, social, pa, gpa):
    raw = {
        "study": study,
        "extra": extra,
        "sleep": sleep,
        "social": social,
        "pa": pa,
        "gpa": gpa,
    }
    return raw, "Collected raw inputs. (Not validated yet.)"

from pydantic import BaseModel, ConfigDict

class OnlineRequired(BaseModel):
    model_config = ConfigDict(strict=True) 

    study: float 
    extra: float   
    sleep: float 
    social: float  
    pa: float     
    gpa: float   


def run_gold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a single-row DataFrame with online inputs and returns Gold features
    by invoking gold_online_row(study, extra, sleep, social, pa, gpa).
    """
    row = df.iloc[0]

    study  = row["study"]
    extra  = row["extra"]
    sleep  = row["sleep"]
    social = row["social"]
    pa     = row["pa"]
    gpa    = row["gpa"]

    sql = """
    SELECT *
    FROM gold_online_row(?, ?, ?, ?, ?, ?)
    """

    with _duck_lock:
        return state.con.execute(sql, [study, extra, sleep, social, pa, gpa]).fetchdf()

def make_one_row_df(payload) -> pd.DataFrame:
    return pd.DataFrame([payload.model_dump()])

def _prepare_X_for_model(X_gold: pd.DataFrame) -> pd.DataFrame:
    cols = list(state.FEATURE_COLUMNS)  
    missing = [c for c in cols if c not in X_gold.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return X_gold.loc[:, cols]

INT_TO_LABEL = ("Low", "Moderate", "High")

def predict_from_raw(study, extra, sleep, social, pa, gpa) -> str:
    raw, _ = collect_raw_inputs(study, extra, sleep, social, pa, gpa)

    try:
        payload = OnlineRequired.model_validate(raw)
    except ValidationError as e:
        raise gr.Error(e.errors()[0]["msg"])
    X_gold = run_gold(make_one_row_df(payload))
    X = _prepare_X_for_model(X_gold)
    yhat = state.MODEL.predict(X)[0]   
    cls_idx = int(np.rint(float(yhat))) 
    cls_idx = int(np.clip(cls_idx, 0, 2)) 
    return INT_TO_LABEL[cls_idx]

def predict(study, extra, sleep, social, pa, gpa):
    return predict_from_raw(study, extra, sleep, social, pa, gpa)

with gr.Blocks() as demo:
    gr.Markdown("## Student Lifestyle")

    with gr.Row():
        with gr.Column():
            study = gr.Slider(minimum=0, maximum=24, step=0.25, value=2.0, label="Study Hours Per Day")
            extra  = gr.Slider(minimum=0, maximum=24, step=0.25, value=1.0, label="Extracurricular Hours Per Day")
            sleep  = gr.Slider(minimum=0, maximum=24, step=0.25, value=8.0, label="Sleep Hours Per Day")

        with gr.Column():
            social = gr.Slider(minimum=0, maximum=24, step=0.25, value=2.0, label="Social Hours Per Day")
            pa     = gr.Slider(minimum=0, maximum=24, step=0.25, value=1.0, label="Physical Activity Hours Per Day")
            gpa    = gr.Slider(minimum=0.0, maximum=4.0, step=0.01, value=3.0, label="GPA")

    submit = gr.Button("Predict Stress Level")
    output = gr.Textbox(label="Predicted Stress Level")

    submit.click(
        fn=predict,
        inputs=[study, extra, sleep, social, pa, gpa],
        outputs=output
    )

if __name__ == "__main__":
    init_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )