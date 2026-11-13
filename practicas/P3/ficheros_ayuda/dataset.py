"""Secuencias largas reproducibles. Los alumnos implementan las ventanas 12 -> 24."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np
from carracing48 import make_env

CONTEXT_LENGTH = 12
PREDICTION_LENGTH = 24
DEFAULT_SEQUENCE_LENGTH = 96


def _driver(env, rng):
    """Política de recogida, nunca entrada del modelo: seguir pista + exploración."""
    car = env.unwrapped.car
    road = np.asarray([(p[2], p[3]) for p in env.unwrapped.track])
    nearest = np.argmin(np.sum((road - np.asarray(car.hull.position))**2, axis=1))
    target = car.hull.GetLocalPoint(tuple(road[(nearest + 5) % len(road)]))
    angle = np.arctan2(target[0], target[1])
    if rng.random() < .25:
        return int(rng.choice(5, p=[.10, .15, .15, .50, .10]))
    if abs(angle) > .15 and rng.random() < .7:
        return 1 if angle > 0 else 2
    return 3


def generate_sequences(seed: int, n_sequences: int = 128, *,
                       sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
                       return_metadata=False, progress=None):
    """Devuelve frames (N,T,48,48,3) uint8 y actions (N,T-1) int64.

    Una secuencia por circuito, sin reinicios internos. action[j] transforma
    frame[j] en frame[j+1]. NO devuelve context/targets: fraccionar las secuencias
    es tarea del alumno. Por defecto T=96, con 61 ventanas posibles 12->24.
    Antes de registrar, descarta U{0,...,200} pasos de observación con acciones
    de la misma política. Cada paso equivale a 4 pasos físicos (0,08 s).
    Reproducible con misma semilla, versiones y plataforma. Separar circuitos
    antes de crear ventanas para evitar fugas entre entrenamiento y validación.
    progress(completed,total), si se proporciona, no recibe semillas.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed debe ser un entero no negativo")
    if isinstance(n_sequences, bool) or not isinstance(n_sequences, int) or n_sequences < 1:
        raise ValueError("n_sequences debe ser un entero positivo")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or not 36 <= sequence_length <= 160:
        raise ValueError("sequence_length debe ser un entero entre 36 y 160")
    rng = np.random.default_rng(seed)
    sequences, all_actions, metadata = [], [], []
    env = make_env(max_episode_steps=200 + sequence_length)
    try:
        attempts = 0
        while len(sequences) < n_sequences:
            attempts += 1
            if attempts > n_sequences * 10:
                raise RuntimeError("No se consiguieron suficientes episodios completos")
            track_seed = int.from_bytes(rng.bytes(16), "big")
            frame, _ = env.reset(seed=track_seed)
            warmup = int(rng.integers(0, 201))
            done = False
            for _ in range(warmup):
                frame, _, term, trunc, _ = env.step(_driver(env, rng))
                if term or trunc:
                    done = True
                    break
            if done:
                continue
            frames, actions = [frame.copy()], []
            for _ in range(sequence_length-1):
                action = _driver(env, rng)
                frame, _, term, trunc, _ = env.step(action)
                frames.append(frame.copy())
                actions.append(action)
                if term or trunc:
                    done = True
                    break
            if done:
                continue
            sequences.append(np.stack(frames))
            all_actions.append(np.asarray(actions, np.int64))
            metadata.append({"track_seed": str(track_seed), "warmup_steps": warmup})
            if progress is not None:
                progress(len(sequences), n_sequences)
    finally:
        env.close()
    result = {"frames": np.stack(sequences), "actions": np.stack(all_actions)}
    return (result, metadata) if return_metadata else result
