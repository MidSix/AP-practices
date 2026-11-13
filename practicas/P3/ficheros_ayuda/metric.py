"""Métrica pública para Kaggle. Menor es mejor; cero corresponde a igualdad."""
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


class ParticipantVisibleError(Exception):
    pass


def visual_metrics(target, prediction):
    """Arrays (...,48,48,3) normalizados a [0,1], con igual peso por frame.

    SSIM RGB con ventana uniforme 7, data_range=1 y covarianza muestral.
    G es MAE de gradientes dividido entre 2 (rango teórico [0,1]).
    Score = .8 * (1 - SSIM)/2 + .2 * G. MSE solo diagnóstico.
    """
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape or target.shape[-3:] != (48,48,3) or target.size == 0:
        raise ValueError("Formas incompatibles: se esperan imágenes (...,48,48,3)")
    for a in (target, prediction):
        if not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
            raise ValueError("Los píxeles deben ser finitos y estar en [0,1]")
    t = target.astype(np.float64).reshape(-1,48,48,3)
    p = prediction.astype(np.float64).reshape(-1,48,48,3)
    ssim = np.mean([structural_similarity(x,y,data_range=1.,channel_axis=-1,
                        win_size=7, gaussian_weights=False, use_sample_covariance=True)
                   for x,y in zip(t,p)])
    gradient = .25 * (np.mean(np.abs(np.diff(t,axis=1)-np.diff(p,axis=1))) +
                      np.mean(np.abs(np.diff(t,axis=2)-np.diff(p,axis=2))))
    return {"score": float(.8*(1-ssim)/2+.2*gradient), "ssim": float(ssim),
            "gradient": float(gradient), "mse": float(np.mean((t-p)**2))}


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Entrada de métrica personalizada Kaggle; soporta subconjuntos públicos/privados."""
    columns = [f"p{i}" for i in range(48*48*3)]
    try:
        if submission.columns.duplicated().any() or solution.columns.duplicated().any():
            raise ValueError("Hay columnas duplicadas")
        if set(submission.columns) != {row_id_column_name, *columns}:
            raise ValueError("Columnas incorrectas")
        for df in (solution, submission):
            if df[row_id_column_name].isna().any() or df[row_id_column_name].duplicated().any():
                raise ValueError("Identificadores vacíos o duplicados")
        if set(solution[row_id_column_name]) != set(submission[row_id_column_name]):
            raise ValueError("Los identificadores no coinciden")
        predicted = submission.set_index(row_id_column_name).loc[solution[row_id_column_name],columns]
        truth = solution[columns].to_numpy(dtype=float)
        if not np.isfinite(truth).all() or (truth < 0).any() or (truth > 255).any() or (truth != np.rint(truth)).any():
            raise ValueError("La solución del anfitrión debe contener enteros 0..255")
        return visual_metrics((truth / 255.).reshape(-1,48,48,3),
                              predicted.to_numpy(dtype=float).reshape(-1,48,48,3))["score"]
    except (ValueError, KeyError, TypeError) as error:
        raise ParticipantVisibleError(str(error)) from error
