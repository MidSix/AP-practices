# P3 · Semana 2 — Autoencoder residual

Autores: Xoel Sánchez Dacoba y Sebastián David Moreno Expósito.

El análisis, los resultados y las conclusiones están en `P3_Semana2_AutoencoderResidual.ipynb` (entregado ya ejecutado). Este fichero solo explica cómo reproducirlo.

## Contenido

| Fichero | Uso |
|---|---|
| `P3_Semana2_AutoencoderResidual.ipynb` | Notebook de la semana 2 |
| `metric.py` | `visual_metrics` (score oficial, SSIM, gradiente, MSE) |
| `dataset.py`, `carracing48.py` | Solo si falta `data/carracing_data.npz`: lo regeneran con las semillas de la semana 1 |
| `requirements.txt` | Dependencias del material de la práctica |
| `data/carracing_data.npz` | Partición de la semana 1: 128 circuitos de entrenamiento (semilla 1488) y 32 de validación (semilla 1314) |
| `models/encoder_d{32,128}.keras`, `models/decoder_d{32,128}.keras` | Autoencoders planos de la semana 1, para la comparación |
| `models/encoder_res_d24.keras`, `models/decoder_res_d24.keras` | Autoencoder residual con saltos (generado) |
| `models/encoder_noskip_d24.keras`, `models/decoder_noskip_d24.keras` | La misma arquitectura sin saltos (generado) |
| `models/encoder_elegido.keras`, `models/decoder_elegido.keras` | Autoencoder elegido y congelado, para las partes 3 y 4 (generado) |
| `models/eleccion.json` | Qué modelo se eligió, con qué criterio y con qué scores |
| `data/latentes_train.npz`, `data/latentes_val.npz` | Latentes por partición: `latents` (N, 96, d) y `actions` (N, 95) (generados) |

## Reproducir

1. Instala las dependencias en un entorno con TensorFlow 2.x y Keras 3:
   ```
   pip install tensorflow -r requirements.txt
   ```
2. Abre el notebook desde esta carpeta y ejecuta todas las celdas. Las rutas son relativas al notebook. Desde la terminal:
   ```
   jupyter nbconvert --to notebook --execute P3_Semana2_AutoencoderResidual.ipynb
   ```
3. Semilla global 77, la misma que en la semana 1, con operaciones deterministas activadas: en la misma máquina, dos ejecuciones dan los mismos resultados. Con GPU, la ejecución completa tarda unos 6 minutos.

## Usar el autoencoder elegido en las partes siguientes

```python
import keras, numpy as np
encoder = keras.models.load_model('models/encoder_elegido.keras')   # (B, 48, 48, 3) -> (B, d), congelado
decoder = keras.models.load_model('models/decoder_elegido.keras')   # (B, d) -> (B, 48, 48, 3) en [0, 1], congelado
lat = np.load('data/latentes_train.npz')                            # lat['latents'], lat['actions']
```
