# Datos y predicciones

## Generación

```python
from dataset import generate_sequences
train = generate_sequences(seed=1234, n_sequences=128)
valid = generate_sequences(seed=5678, n_sequences=32)
```

Cada resultado contiene `frames` de forma `(N,96,48,48,3)` (uint8) y
`actions` de forma `(N,95)` (enteros). Normaliza las imágenes dividiendo entre
255. La acción `actions[i,j]` lleva de `frames[i,j]` a `frames[i,j+1]`.
Antes de registrar cada secuencia se ejecutan entre 0 y 200 pasos elegidos al azar.
Guarda los datos y las semillas para reproducir los experimentos.

Acciones: 0 nada, 1 derecha, 2 izquierda, 3 acelerar, 4 frenar.
Cada acción dura 0,08 s simulados. Separa los circuitos de entrenamiento y
validación antes de construir ventanas. Cada ventana debe contener 12 imágenes
de contexto, 24 objetivos y 35 acciones (11 históricas y 24 futuras).

## Entradas de Kaggle

```python
import numpy as np
inputs = np.load('test_inputs.npz', allow_pickle=False)
context = inputs['context'].astype('float32') / 255
actions = inputs['actions']
```

`context` tiene forma `(32,12,48,48,3)`, `actions` tiene forma `(32,35)` y
`sequence_id` contiene los 32 identificadores. No se incluyen imágenes objetivo.
No uses estas entradas para entrenar ni seleccionar hiperparámetros.

## Envío

Una vez calculadas tus predicciones de forma `(32,24,48,48,3)`, en `[0,1]`:

```python
from submission import write_submission
write_submission(predictions, inputs['sequence_id'], 'submission.csv')
```

El archivo contiene 768 filas con `id` y las columnas `p0` a `p6911`.
Los píxeles siguen el orden fila, columna y canal RGB. Conserva los identificadores.

## Evaluación local

```python
from metric import visual_metrics
result = visual_metrics(objetivos, predicciones)
```

Ambos arrays deben tener la misma forma y píxeles finitos en `[0,1]`.
La función devuelve `score`, `ssim`, `gradient` y `mse`. Menor `score` es mejor.
La función `score` del mismo módulo es el evaluador de CSV utilizado por Kaggle.
