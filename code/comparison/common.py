"""Utilidades compartidas de las tareas de comparación (consola, CSV, metadata)."""
from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import entropy, mannwhitneyu

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


_METADATA_BUFFER: list[tuple] = []


def take_metadata() -> list[tuple]:
    """Extrae las filas de metadata pendientes (para serializar en el padre)."""
    rows = list(_METADATA_BUFFER)
    _METADATA_BUFFER.clear()
    return rows


def replay_metadata(rows: list[tuple]) -> None:
    """Escribe en el padre las filas capturadas en los workers."""
    for args in rows:
        _append_metadata(*args)


def append_metadata(
    metadata_path: Path, task: str, params: dict[str, Any], repo_dir: Path
) -> Path:
    """Añade una fila por corrida/tarea a run_metadata.csv (reescritura segura del BOM)."""
    import os

    if os.environ.get("PERF2_BUFFER_METADATA"):
        _METADATA_BUFFER.append((metadata_path, task, params, repo_dir))
        return Path(metadata_path)
    return _append_metadata(metadata_path, task, params, repo_dir)


def _append_metadata(
    metadata_path: Path, task: str, params: dict[str, Any], repo_dir: Path
) -> Path:
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


# ───────────────────── Estadística compartida (sin copias locales) ─────────────────────
def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg (q=0.05 se aplica al veredicto, no al ajuste)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


def _bootstrap_counts(labels: np.ndarray, n: int, b: int, t: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Conteos de tópico (b,t) mediante indexado fancy en bloque, sin loop por réplica."""
    idx = rng.integers(0, n, size=(b, n))
    flat = np.asarray(labels, dtype=int)[idx] + 1 + np.arange(b)[:, None] * (t + 1)
    return np.bincount(flat.ravel(), minlength=b * (t + 1)).reshape(b, t + 1)[:, 1:] / n


def bootstrap_gap_ci(lab_d: np.ndarray, lab_u: np.ndarray, T: int,
                     B: int, seed: int) -> np.ndarray:
    """CI 95% de gap_pp por tópico; remuestreo independiente dentro de cada corpus."""
    nd, nu = len(lab_d), len(lab_u)
    if B <= 0 or nd == 0 or nu == 0:
        return np.full((2, T), np.nan)
    rng = np.random.default_rng(seed)
    gaps = np.empty((B, T))
    for start in range(0, B, 256):
        b = min(256, B - start)
        gaps[start:start + b] = 100 * (
            _bootstrap_counts(lab_d, nd, b, T, rng)
            - _bootstrap_counts(lab_u, nu, b, T, rng))
    return np.percentile(gaps, [2.5, 97.5], axis=0)


def bootstrap_rate_ci(mask: np.ndarray, b: int, seed: int) -> tuple[float, float]:
    """CI 95% bootstrap (percentiles) de una tasa binaria (función pura)."""
    x = np.asarray(mask, dtype=float)
    if not x.size or b <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(int(b), x.size))
    rates = x[idx].mean(axis=1)
    return (float(np.quantile(rates, 0.025)), float(np.quantile(rates, 0.975)))


def jsd_base2(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence base 2 (rango [0,1])."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2)))


def mann_whitney_cliffs(x: Sequence[float], y: Sequence[float]) -> dict:
    """Mann-Whitney U + Cliff's delta con interpretación (Romano et al.)."""
    a = np.asarray(x, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(y, dtype=float); b = b[~np.isnan(b)]
    if not a.size or not b.size:
        return {"mw_u": np.nan, "mw_p_value": np.nan,
                "cliffs_delta": np.nan, "cliffs_interp": "SIN_DATOS"}
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    delta = 2.0 * float(u) / (a.size * b.size) - 1.0
    ad = abs(delta)
    interp = ("NEGLIGIBLE" if ad < 0.147 else "PEQUENO" if ad < 0.33
              else "MEDIANO" if ad < 0.474 else "GRANDE")
    return {"mw_u": round(float(u), 4), "mw_p_value": round(float(p), 6),
            "cliffs_delta": round(delta, 6), "cliffs_interp": interp}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Normaliza L2 por fila con suelo numérico."""
    x = np.asarray(x, dtype=float)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def max_cosine(docs: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Coseno por fila y máximo sobre refs. docs (N,D), refs (M,D)."""
    docs, refs = np.atleast_2d(docs), np.atleast_2d(refs)
    if refs.shape[0] == 0:
        return np.full(docs.shape[0], np.nan, dtype=float)
    return np.asarray(l2_normalize(docs) @ l2_normalize(refs).T).max(axis=1)