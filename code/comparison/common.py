"""Utilidades compartidas de las tareas de comparación (consola, CSV, metadata)."""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEP = "█" * 70
CSV_ENCODING = "utf-8-sig"


def print_header(title: str) -> None:
    """Imprime un encabezado con el estilo del pipeline."""
    print(f"\n{SEP}\n   {title}\n{SEP}")


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    """Guarda un DataFrame como CSV utf-8-sig, index=False (regla del proyecto)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)
    return path


def git_hash(repo_dir: Path) -> str:
    """Devuelve el hash HEAD del repo, o 'no_git' si no está disponible."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "no_git"
    except Exception:
        return "no_git"


def library_versions() -> dict[str, str]:
    """Versiones de las librerías científicas relevantes."""
    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    for name in ("sklearn", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = "no_disponible"
    return versions


def append_metadata(
    metadata_path: Path, task: str, params: dict[str, Any], repo_dir: Path
) -> Path:
    """Añade una fila por corrida/tarea a run_metadata.csv (reescritura segura del BOM)."""
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "task": task,
        "params": json.dumps(params, ensure_ascii=False, sort_keys=True, default=str),
        "git_hash": git_hash(repo_dir),
        **{f"version_{k}": v for k, v in library_versions().items()},
    }
    new = pd.DataFrame([row])
    metadata_path = Path(metadata_path)
    if metadata_path.exists():
        old = pd.read_csv(metadata_path, encoding=CSV_ENCODING)
        new = pd.concat([old, new], ignore_index=True)
    return save_csv(new, metadata_path)


def palette_colors(palette: Any, fallback: str = "#4C72B0") -> list[str]:
    """Normaliza la paleta a una lista de colores CSS indexable por posición.

    utils.palette_to_plot_colors puede devolver un dict (categoría→color) o una
    lista; los gráficos de comparación necesitan acceso posicional (colors[i]).
    """
    if palette is None:
        return [fallback]
    vals = list(palette.values()) if isinstance(palette, dict) else (
        list(palette) if isinstance(palette, (list, tuple)) else [palette])
    out: list[str] = []
    for v in vals:
        if isinstance(v, (list, tuple)):
            out.extend(str(c) for c in v)
        elif isinstance(v, dict):
            out.extend(str(c) for c in v.values())
        elif v is not None:
            out.append(str(v))
    return out or [fallback]