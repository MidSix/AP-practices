#!/usr/bin/env python3
"""Script para generar y congelar el conjunto de datos de CarRacing (Semana 1 a 5).

Uso:
    python generate_data.py
    python generate_data.py --train_seqs 128 --val_seqs 32 --train_seed 1234 --val_seed 5678
"""

import os
import sys
import argparse
import time
import numpy as np

# Asegurar que el directorio actual está en el sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset import generate_sequences


def main():
    parser = argparse.ArgumentParser(description="Generar y guardar datos reproducibles de CarRacing.")
    parser.add_argument("--train_seed", type=int, default=1234, help="Semilla para secuencias de entrenamiento.")
    parser.add_argument("--val_seed", type=int, default=5678, help="Semilla para secuencias de validación.")
    parser.add_argument("--train_seqs", type=int, default=128, help="Número de secuencias de entrenamiento.")
    parser.add_argument("--val_seqs", type=int, default=32, help="Número de secuencias de validación.")
    parser.add_argument("--output", type=str, default="data/carracing_data.npz", help="Ruta de guardado .npz.")
    args = parser.parse_args()

    output_path = os.path.join(SCRIPT_DIR, args.output) if not os.path.isabs(args.output) else args.output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print(" GENERACIÓN DE DATASET DE CARRACING - PRÁCTICA 3")
    print("=" * 60)
    print(f"Entrenamiento: {args.train_seqs} secuencias | Semilla: {args.train_seed}")
    print(f"Validación:    {args.val_seqs} secuencias | Semilla: {args.val_seed}")
    print(f"Destino:       {output_path}")
    print("=" * 60)

    def progress_callback(prefix):
        def _cb(completed, total):
            pct = 100.0 * completed / total
            bar_len = 30
            filled = int(bar_len * completed / total)
            bar = "=" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r{prefix} [{bar}] {completed}/{total} ({pct:.1f}%)")
            sys.stdout.flush()
        return _cb

    # Generación de entrenamiento
    t0 = time.time()
    print("\n[1/2] Generando secuencias de entrenamiento...")
    train_result = generate_sequences(
        seed=args.train_seed,
        n_sequences=args.train_seqs,
        progress=progress_callback("Train:")
    )
    print(f"\nEntrenamiento generado en {time.time() - t0:.1f} segundos.")

    # Generación de validación
    t1 = time.time()
    print("\n[2/2] Generando secuencias de validación...")
    val_result = generate_sequences(
        seed=args.val_seed,
        n_sequences=args.val_seqs,
        progress=progress_callback("Val:  ")
    )
    print(f"\nValidación generada en {time.time() - t1:.1f} segundos.")

    # Guardar en formato comprimido .npz
    print(f"\nGuardando datos comprimidos en {output_path}...")
    np.savez_compressed(
        output_path,
        train_frames=train_result["frames"],
        train_actions=train_result["actions"],
        val_frames=val_result["frames"],
        val_actions=val_result["actions"],
        train_seed=args.train_seed,
        val_seed=args.val_seed,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"¡Guardado completado exitosamente! Tamaño del archivo: {size_mb:.2f} MB")
    print(f"Train frames: {train_result['frames'].shape}, dtype={train_result['frames'].dtype}")
    print(f"Train actions: {train_result['actions'].shape}, dtype={train_result['actions'].dtype}")
    print(f"Val frames:   {val_result['frames'].shape}, dtype={val_result['frames'].dtype}")
    print(f"Val actions:   {val_result['actions'].shape}, dtype={val_result['actions'].dtype}")
    print("=" * 60)


if __name__ == "__main__":
    main()

