"""Exportador público de predicciones (N,24,48,48,3) en [0,1]."""
import numpy as np
import pandas as pd


def write_submission(predictions, sequence_ids, path):
    predictions = np.asarray(predictions)
    sequence_ids = np.asarray(sequence_ids)
    if predictions.shape != (len(sequence_ids),24,48,48,3):
        raise ValueError("Se espera (N,24,48,48,3)")
    if len(set(sequence_ids.tolist())) != len(sequence_ids):
        raise ValueError("IDs de secuencia duplicados")
    if not np.isfinite(predictions).all() or (predictions<0).any() or (predictions>1).any():
        raise ValueError("Predicciones fuera de [0,1] o no finitas")
    rows = [f"{sid}_{h}" for sid in sequence_ids for h in range(1,25)]
    df = pd.DataFrame(predictions.reshape(-1,6912), columns=[f"p{i}" for i in range(6912)])
    df.insert(0,"id",rows)
    df.to_csv(path,index=False,float_format="%.9g")
    return df
