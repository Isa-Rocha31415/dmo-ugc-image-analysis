"""
metrics_lexical.py — TAREA P2: capa léxica L1 (DMO vs UGC agregado y por fuente).

- Tokenización unicode, minúsculas, alfabética len>=3, SIN listas de stopwords.
  El control de términos ubicuos/raros se hace con filtros estadísticos del propio
  corpus combinado (min_df / max_df), no con recursos externos.
- Lematización solo si spaCy está disponible offline (si no, se omite y se registra).
- Jaccard@k (k=50,100,200) por frecuencia y por TF-IDF; cobertura de tipos (freq>=2)
  y de tokens; bootstrap CI 95% (B=--bootstrap) para Jaccard@100 y coberturas.
- Log-odds con prior informativo de Dirichlet (Monroe, Colaresi & Quinn 2008):
    alpha_w = alpha0 * p_pool(w)
    delta_w = log((y_d+alpha)/(n_d+alpha0-y_d-alpha)) - log((y_u+alpha)/(n_u+alpha0-y_u-alpha))
    var(delta) ≈ 1/(y_d+alpha) + 1/(y_u+alpha);  z = delta / sqrt(var)

Outputs en ../results/comparison/lexical/:
  metrics_lexical.csv, logodds_dmo_vs_ugc.csv, logodds_by_source.csv,
  wordcloud_logodds_dmo.html (+_data.csv), wordcloud_logodds_ugc.html (+_data.csv)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from comparison.common import append_metadata, print_header, save_csv
from io_corpus import CORPUS_COLUMN, SOURCE_COLUMN
from utils import palette_to_plot_colors

TOKEN_RE = re.compile(r"(?u)\b[^\W\d_]{3,}\b")
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"
KS = (50, 100, 200)
MIN_TYPE_FREQ = 2
LEXICAL_CACHE_VERSION = 2  # súbelo si cambia la tokenización o build_matrix


# ─────────────────────────── Tokenización (sin stopwords) ───────────────────────────
def tokenize(text: str) -> list[str]:
    """Tokens unicode alfabéticos de longitud>=3, en minúsculas. SIN stopwords.

    No se usa ninguna lista externa: el control de palabras funcionales se hace
    después con el filtro estadístico max_df sobre el corpus combinado.
    """
    return TOKEN_RE.findall(str(text).lower())


def try_spacy() -> tuple[Optional[object], str]:
    """Carga spaCy SOLO si ya está instalado offline; nunca descarga."""
    try:
        import spacy
        for name in ("es_core_news_sm", "es_core_news_md", "es_core_news_lg"):
            try:
                return spacy.load(name, disable=["parser", "ner"]), name
            except OSError:
                continue
    except Exception:
        pass
    return None, "none"


def tokenize_corpus(texts: list[str], nlp=None) -> list[str]:
    """Textos tokenizados (lematizados si hay spaCy), unidos por espacios."""
    if nlp is None:
        return [" ".join(tokenize(t)) for t in texts]
    out = []
    for doc in nlp.pipe(texts, batch_size=500):
        out.append(" ".join(
            tok.lemma_.lower() for tok in doc
            if tok.is_alpha and len(tok.lemma_) >= 3
        ))
    return out


def build_matrix(joined: list[str], min_df: int = 5,
                 max_df: float = 0.40) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Matriz documento-término; relaja filtros si el vocabulario queda vacío.

    min_df/max_df se estiman del corpus COMBINADO (DMO+UGC), no por grupo,
    para que ambos lados compartan exactamente el mismo vocabulario.
    """
    intentos = [(min_df, max_df),
                (max(1, min_df // 2), min(1.0, max_df + 0.2)),
                (1, 1.0)]
    for md, xd in intentos:
        try:
            vec = CountVectorizer(token_pattern=TOKEN_PATTERN,
                                  lowercase=False, min_df=md, max_df=xd)
            X = vec.fit_transform(joined).astype(np.int64)
            if X.shape[1] == 0:
                raise ValueError("empty vocabulary")
            if (md, xd) != (min_df, max_df):
                print(f"  ⚠️ Vocabulario vacío con min_df={min_df}/max_df={max_df}; "
                      f"relajado a min_df={md}/max_df={xd} (corpus pequeño).")
            return X, np.asarray(vec.get_feature_names_out())
        except ValueError:
            continue
    raise ValueError("No se pudo construir vocabulario: corpus vacío o demasiado pequeño.")


# ─────────────────────────── Métricas puras ───────────────────────────
def _topk(scores: np.ndarray, k: int) -> set[int]:
    """Top-k términos con score ESTRICTAMENTE positivo (nunca rellena con ceros)."""
    idx = np.flatnonzero(scores > 0)
    if idx.size == 0:
        return set()
    k = min(k, idx.size)
    s = scores[idx]
    sel = np.argpartition(-s, k - 1)[:k]
    return set(idx[sel].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    """|A∩B| / |A∪B|."""
    return len(a & b) / max(1, len(a | b))


def tfidf_scores(freq: np.ndarray, df: np.ndarray, n: int) -> np.ndarray:
    """TF agregado × IDF suavizado del corpus (o del remuestreo)."""
    return freq * (np.log((1.0 + n) / (1.0 + df)) + 1.0)


def coverages(fd: np.ndarray, fu: np.ndarray) -> tuple[float, float]:
    """(cobertura de tipos freq>=2, cobertura de tokens DMO presentes en UGC)."""
    vd, vu = fd >= MIN_TYPE_FREQ, fu >= MIN_TYPE_FREQ
    cov_types = float((vd & vu).sum() / max(1, vd.sum()))
    mask = fu > 0
    cov_tokens = float(fd[mask].sum() / max(1, fd.sum()))
    return cov_types, cov_tokens


def logodds_z(fd: np.ndarray, fu: np.ndarray, alpha0: float = 1000.0) -> np.ndarray:
    """Log-odds con prior dirichlet informativo (Monroe et al. 2008).

    Términos con pooled==0 reciben z=0 (se filtran aguas abajo con `keep`).
    """
    pooled = fd + fu
    total = float(pooled.sum())
    z = np.zeros_like(pooled, dtype=float)
    ok = pooled > 0
    if not ok.any() or total <= 0:
        return z
    a = alpha0 * pooled[ok] / total
    f_d, f_u = fd[ok], fu[ok]
    nd, nu = fd.sum(), fu.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        od = (f_d + a) / np.maximum(nd + alpha0 - f_d - a, 1e-12)
        ou = (f_u + a) / np.maximum(nu + alpha0 - f_u - a, 1e-12)
        delta = np.log(od) - np.log(ou)
        var = 1.0 / (f_d + a) + 1.0 / (f_u + a)
        z[ok] = delta / np.sqrt(var)
    return z


def bootstrap_lexical(
    Xd: sparse.csr_matrix, Xu: sparse.csr_matrix, B: int, seed: int
) -> dict[str, tuple[float, float]]:
    """CI 95% bootstrap exacto por remuestreo de documentos.

    Equivalencia: el conteo de apariciones de cada documento en un remuestreo
    con reemplazo de tamaño n es c ~ Multinomial(n, 1/n). Por tanto
    freq_remuestreada = Xᵀ·c  y  df_remuestreada = Bᵀ·1[c>0].
    Se implementa como producto disperso matriz-vector (10–50× más rápido que
    rebanar filas), con la misma distribución y reproducible vía --seed.
    """
    metrics = ("jaccard_freq", "jaccard_tfidf", "coverage_types", "coverage_tokens")
    nd, nu = Xd.shape[0], Xu.shape[0]
    if B <= 0 or nd == 0 or nu == 0:
        return {m: (np.nan, np.nan) for m in metrics}

    Bd, Bu = Xd.copy(), Xu.copy()
    Bd.data[:] = 1
    Bu.data[:] = 1
    XdT, XuT = Xd.T.tocsr(), Xu.T.tocsr()
    BdT, BuT = Bd.T.tocsr(), Bu.T.tocsr()
    p_d, p_u = np.full(nd, 1.0 / nd), np.full(nu, 1.0 / nu)

    rng = np.random.default_rng(seed)
    acc = {m: [] for m in metrics}
    for _ in range(B):
        cd = rng.multinomial(nd, p_d).astype(np.float64)
        cu = rng.multinomial(nu, p_u).astype(np.float64)
        fd, fu = XdT @ cd, XuT @ cu
        acc["jaccard_freq"].append(jaccard(_topk(fd, 100), _topk(fu, 100)))
        sd = tfidf_scores(fd, BdT @ (cd > 0).astype(np.float64), nd)
        su = tfidf_scores(fu, BuT @ (cu > 0).astype(np.float64), nu)
        acc["jaccard_tfidf"].append(jaccard(_topk(sd, 100), _topk(su, 100)))
        ct, ck = coverages(fd, fu)
        acc["coverage_types"].append(ct)
        acc["coverage_tokens"].append(ck)
    return {m: tuple(np.percentile(v, [2.5, 97.5])) for m, v in acc.items()}


def interpret_lexical(j100: float, cov_types: float) -> str:
    """Frase interpretativa automática (umbrales documentados)."""
    v = "bajo" if j100 < 0.25 else ("moderado" if j100 < 0.50 else "alto")
    c = "apenas aparece en el UGC" if cov_types < 0.50 else "está mayormente presente en el UGC"
    return (f"Vocabulario compartido {v} (Jaccard@100={j100:.2f}); "
            f"el vocabulario distintivo de la DMO {c} (cobertura de tipos={cov_types:.2f}).")


def _wordcloud_html(terms: pd.DataFrame, path: Path, title: str, colors: list[str]) -> Path:
    """Wordcloud plotly (tamaño=|z|, layout filotáctico determinista) + CSV de datos.

    Devuelve la ruta del HTML; el CSV de datos se escribe junto a él con el
    sufijo _data.csv (regla del proyecto: toda figura exporta su tabla).
    """
    import plotly.graph_objects as go

    terms = terms.reset_index(drop=True)
    n = len(terms)
    data_path = path.with_name(path.stem + "_data.csv")

    if n == 0:
        fig = go.Figure()
        fig.update_layout(title=f"{title} — sin términos",
                          template="plotly_white", height=400)
        fig.write_html(path, include_plotlyjs=True)
        save_csv(pd.DataFrame(columns=["term", "logodds_z", "size", "x", "y"]), data_path)
        return path

    z = terms["logodds_z"].abs().to_numpy()
    size = 12 + 38 * (z - z.min()) / max(1e-9, z.max() - z.min())
    theta = 2.399963 * np.arange(n)          # ángulo áureo (filotaxis)
    r = np.sqrt(np.arange(n) + 1)
    x, y = 2.6 * r * np.cos(theta), r * np.sin(theta)   # 2.6 reduce solapamiento

    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="text", text=terms["term"],
        textfont=dict(size=size, color=[colors[i % len(colors)] for i in range(n)]),
        hovertemplate="%{text}<br>z=%{customdata:.2f}<extra></extra>",
        customdata=terms["logodds_z"],
    ))
    fig.update_layout(title=title, template="plotly_white", height=700,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.write_html(path, include_plotlyjs=True)
    save_csv(pd.DataFrame({"term": terms["term"], "logodds_z": terms["logodds_z"],
                           "size": size, "x": x, "y": y}), data_path)
    return path


# ─────────────────────────── Tarea 'lexical' ───────────────────────────
def run_lexical_task(args: argparse.Namespace, results_dir: Path) -> list[Path]:
    """Tarea P2: capa léxica L1 con CI bootstrap y log-odds con prior."""
    print_header("PASO 10: CAPA LÉXICA L1 — JACCARD, COBERTURA Y LOG-ODDS (DMO vs UGC)")
    base_dir = results_dir / "comparison" / "base"
    out_dir = results_dir / "comparison" / "lexical"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_dir / "analysis_base.csv", encoding="utf-8-sig")
    texts = base["text"].fillna("").astype(str).tolist()
    corpus = base[CORPUS_COLUMN].astype(str).to_numpy()
    source = base[SOURCE_COLUMN].astype(str).to_numpy()
    print(f"  🔤 Vocabulario: token alfabético len>=3, min_df={args.min_df}, "
          f"max_df={args.max_df} (sin listas de stopwords)")

    nlp, lemma_model = (None, "none") if args.disable_lemma else try_spacy()
    print(f"  Lematización: "
          f"{'spaCy ' + lemma_model if nlp else 'omitida (spaCy no disponible offline)'}")

    # ── Caché propia de la matriz documento-término ──
    key = hashlib.sha256(json.dumps({
        "texts_sha": hashlib.sha256("\n".join(texts).encode()).hexdigest(),
        "lemma": lemma_model, "min_df": args.min_df, "max_df": args.max_df,
        "version": LEXICAL_CACHE_VERSION,
    }, sort_keys=True).encode()).hexdigest()[:20]
    cache = out_dir / "_cache"
    cache.mkdir(exist_ok=True)
    npz = cache / f"docterm_{key}.npz"
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        X = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=d["shape"])
        vocab = d["vocab"]
        print(f"  ✅ Caché hit [lexical/docterm] -> {npz}")
    else:
        joined = tokenize_corpus(texts, nlp)
        X, vocab = build_matrix(joined, min_df=args.min_df, max_df=args.max_df)
        np.savez(npz, data=X.data, indices=X.indices, indptr=X.indptr,
                 shape=X.shape, vocab=vocab)
    print(f"  📐 Matriz documento-término: {X.shape[0]:,} docs × {X.shape[1]:,} tipos")

    # ── Lado DMO (fijo para todos los scopes) ──
    dmo_mask = corpus == "DMO"
    Xd = X[dmo_mask]
    if Xd.shape[0] == 0:
        raise ValueError("No hay documentos DMO en analysis_base; revisa --dmo-label.")
    fd_full = np.asarray(Xd.sum(axis=0)).ravel().astype(float)
    dfd = np.asarray((Xd > 0).sum(axis=0)).ravel()        # una sola vez
    sd_full = tfidf_scores(fd_full, dfd, Xd.shape[0])     # TF-IDF DMO, una sola vez

    scopes = [("ugc_total", corpus == "UGC")] + [
        (s, (corpus == "UGC") & (source == s))
        for s in sorted(set(source[corpus == "UGC"]))
    ]

    rows, logodds_src = [], []
    agg_logodds: Optional[pd.DataFrame] = None
    colors = palette_to_plot_colors(args.palette)
    written: list[Path] = []

    for scope, mask in scopes:
        Xu = X[mask]
        nd, nu = Xd.shape[0], Xu.shape[0]
        if nu == 0:
            print(f"  ⚠️ scope={scope} sin documentos; se omite.")
            continue
        fu_full = np.asarray(Xu.sum(axis=0)).ravel().astype(float)
        dfu = np.asarray((Xu > 0).sum(axis=0)).ravel()
        su_full = tfidf_scores(fu_full, dfu, nu)

        print(f"\n  🧮 scope={scope} | n_DMO={nd:,} n_UGC={nu:,} | bootstrap B={args.bootstrap}...")
        ci = bootstrap_lexical(Xd, Xu, args.bootstrap, args.seed)

        for k in KS:
            jf = jaccard(_topk(fd_full, k), _topk(fu_full, k))
            jt = jaccard(_topk(sd_full, k), _topk(su_full, k))
            for metric, val in (("jaccard_freq", jf), ("jaccard_tfidf", jt)):
                lo, hi = ci[metric] if k == 100 else (np.nan, np.nan)
                rows.append(dict(metric=metric, scope=scope, k=k, value=round(val, 6),
                                 ci_low=lo, ci_high=hi, n_dmo=nd, n_ugc=nu))

        ct, ck = coverages(fd_full, fu_full)
        for metric, val in (("coverage_types", ct), ("coverage_tokens", ck)):
            lo, hi = ci[metric]
            rows.append(dict(metric=metric, scope=scope, k=np.nan, value=round(val, 6),
                             ci_low=lo, ci_high=hi, n_dmo=nd, n_ugc=nu))

        z = logodds_z(fd_full, fu_full, args.prior_strength)
        keep = (fd_full + fu_full) > 0
        lo_df = pd.DataFrame({
            "term": vocab[keep],
            "freq_dmo": fd_full[keep].astype(int),
            "freq_ugc": fu_full[keep].astype(int),
            "logodds_z": z[keep],
            "direction": np.where(z[keep] > 0, "DMO", "UGC"),
        }).sort_values("logodds_z", ascending=False)

        if scope == "ugc_total":
            agg_logodds = lo_df
            written.append(save_csv(lo_df, out_dir / "logodds_dmo_vs_ugc.csv"))
        else:
            logodds_src.append(lo_df.assign(source=scope))

    if agg_logodds is None:
        raise ValueError("No se pudo calcular el agregado UGC (scope 'ugc_total' vacío).")

    written.insert(0, save_csv(pd.DataFrame(rows), out_dir / "metrics_lexical.csv"))

    # ── log-odds por fuente (siempre se escribe, aunque no haya fuentes) ──
    if logodds_src:
        lo_src = pd.concat(logodds_src, ignore_index=True)
    else:
        lo_src = pd.DataFrame(columns=["term", "freq_dmo", "freq_ugc",
                                       "logodds_z", "direction", "source"])
    written.append(save_csv(lo_src, out_dir / "logodds_by_source.csv"))

    # ── Wordclouds por dirección (top-60 |z|, signo original conservado) ──
    top_dmo = agg_logodds[agg_logodds.direction == "DMO"].nlargest(60, "logodds_z")
    top_ugc = agg_logodds[agg_logodds.direction == "UGC"].nsmallest(60, "logodds_z")
    for fname, dfw, titulo in (
        ("wordcloud_logodds_dmo.html", top_dmo,
         "Términos distintivos DMO (log-odds z, Monroe 2008)"),
        ("wordcloud_logodds_ugc.html", top_ugc,
         "Términos distintivos UGC (log-odds z)"),
    ):
        p = _wordcloud_html(dfw, out_dir / fname, titulo, colors)
        written.append(p)
        written.append(p.with_name(p.stem + "_data.csv"))

    # ── Consola ──
    m = pd.DataFrame(rows)
    mu = m[m.scope == "ugc_total"]

    def _val(metric: str, k=None):
        sel = mu[mu.metric == metric]
        sel = sel[sel.k.isna()] if k is None else sel[sel.k == k]
        return sel.iloc[0]

    j100 = float(_val("jaccard_freq", 100).value)
    ct = float(_val("coverage_types").value)
    print("\n  📊 DMO vs UGC_total:")
    for metric, k in (("jaccard_freq", 100), ("jaccard_tfidf", 100),
                      ("coverage_types", None), ("coverage_tokens", None)):
        r = _val(metric, k)
        print(f"    {metric:<18s} = {r.value:.3f}  CI95% [{r.ci_low:.3f}, {r.ci_high:.3f}]")

    print("\n  🥇 Top-20 distintivos DMO (z):")
    for _, r in agg_logodds.head(20).iterrows():
        print(f"    {r.term:<22s} z={r.logodds_z:+7.2f}  (DMO={r.freq_dmo}, UGC={r.freq_ugc})")
    print("\n  🥇 Top-20 distintivos UGC (z):")
    for _, r in agg_logodds.tail(20).iloc[::-1].iterrows():
        print(f"    {r.term:<22s} z={r.logodds_z:+7.2f}  (DMO={r.freq_dmo}, UGC={r.freq_ugc})")
    print(f"\n  💬 {interpret_lexical(j100, ct)}")
    print("  ℹ️ Jaccard por frecuencia incluye palabras funcionales (sin stopwords); "
          "la lectura sustantiva es jaccard_tfidf.")

    written.append(append_metadata(
        base_dir / "run_metadata.csv", task="lexical",
        params={**vars(args), "lemma_model": lemma_model,
                "prior_strength": args.prior_strength,
                "min_type_freq": MIN_TYPE_FREQ, "ks": list(KS),
                "n_types": int(X.shape[1]),
                "stopwords": "none (solo filtros min_df/max_df del corpus)"},
        repo_dir=Path(__file__).resolve().parent.parent,
    ))
    return written