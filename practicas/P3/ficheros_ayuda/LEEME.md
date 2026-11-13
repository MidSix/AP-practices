# Material de la práctica

Extrae los archivos en la misma carpeta que tus notebooks.

```python
%pip install -r requirements.txt
```

Si Box2D necesita compilarse, instala antes `swig` con `%pip install swig`.
Para el juego, ejecuta `%pip install -r requirements-interactive.txt`.
Necesitas también Keras y su backend.

- `dataset.py`: genera secuencias de entrenamiento y validación.
- `carracing48.py`: entorno con imágenes de 48 × 48 píxeles.
- `metric.py`: evaluación de imágenes y envíos.
- `submission.py`: exportación del CSV para Kaggle.
- `interactive_alumno.py`: clase del juego que debes completar.
- `comparador.ipynb`: ejecución del juego con tus modelos.
- `test_inputs.npz`: entradas de evaluación.
- `sample_submission.csv`: ejemplo que repite la última imagen observada.

Consulta `DATOS.md` para generar datos y preparar el envío, e `INTERACTIVO.md`
para conectar los modelos al juego. Las tareas y reglas se indican en el enunciado.
