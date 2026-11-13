# Juego con tus modelos

Completa `_initialize_latents`, `_predict_next` y `_advance_history` de
`ComparisonSession` en `interactive_alumno.py`. Sus docstrings indican las
entradas y salidas. Lee también `reset` y `step`. Los controles y paneles ya están
implementados; los métodos pendientes lanzan `NotImplementedError`.

Modelos requeridos:

- Encoder: imágenes `(B,48,48,3)` → latentes `(B,D)`.
- Predictor: lista `[latentes (B,12,D), acciones históricas (B,11), acción (B,)]`
  → siguiente latente `(B,D)`.
- Decoder: latentes `(B,D)` → imágenes `(B,48,48,3)` en `[0,1]`.

Usa acciones enteras 0–4 y modelos con `stateful=False`. Codifica las doce
imágenes iniciales una sola vez. Después realimenta los latentes predichos;
no vuelvas a codificar las imágenes generadas ni uses las nuevas imágenes reales.

Abre `comparador.ipynb`, instala las dependencias y ajusta las tres rutas.
Para capas personalizadas, impórtalas antes de cargar o pasa `custom_objects`.

Pulsa Jugar y mantén el foco en el campo del teclado: W acelera, S frena y A/D
giran. Sin teclas se aplica nada; con varias, manda la última mantenida.
Escape o perder el foco pausa. El objetivo es 24 Hz; la frecuencia efectiva
puede ser menor. Cada paso conserva 0,08 s simulados (a 24 Hz, velocidad 1,92×).

Los botones permiten avanzar paso a paso, reiniciar, reproducir acciones y evaluar
la partida. Cierra la aplicación con `app.close()` antes de crear otra.
Tras modificar la clase, reinicia el kernel y ejecuta de nuevo las celdas.
