import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_colors_from_cmap(cmap_name, values):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i) for i in values]


def abbreviate_basis_set(basis_set):
    abbreviations = {
        "cc-pVDZ": "DZ",
        "cc-pVTZ": "TZ",
        "cc-pVQZ": "QZ",
        "cc-pV5Z": "5Z",
        "def2-SVP": "DZ",
        "def2-TZVP": "TZ",
        "def2-QZVP": "QZ",
    }
    for k, v in abbreviations.items():
        basis_set = basis_set.replace(k, v)
    return basis_set


def get_outlier_mask(x, window_size=1000, quantile=0.01, outlier_range=5):
    is_outlier = np.zeros(len(x), dtype=bool)
    if not window_size:
        window_size = len(x)
    window_size = min(window_size, len(x))
    for idx_block in range(int(np.ceil(len(x) / window_size))):
        idx_start = min(idx_block * window_size, len(x) - window_size)
        idx_end = idx_start + window_size
        x_window = x[idx_start:idx_end]
        if np.isfinite(x_window).sum() == 0:
            is_outlier[idx_start:idx_end] = True
            continue
        qlow = x_window.quantile(quantile)
        qhigh = x_window.quantile(1 - quantile)
        med = x_window.median()
        included_range = outlier_range * (qhigh - qlow)
        is_outlier[idx_start:idx_end] = (x_window < med - included_range) | (x_window > med + included_range)
    return is_outlier


def savefig(fig, name, pdf=True, png=True, bbox_inches="tight"):
    if name.endswith(".png") or name.endswith(".pdf"):
        name = name[:-4]
    if png:
        fig.savefig(name + ".png", bbox_inches=bbox_inches, dpi=200)
    if pdf:
        fig.savefig(name + ".pdf", bbox_inches=bbox_inches)


COLOR_FIRE = "#e15759"
COLOR_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#9c755f",
    # "#e15759", # used for FiRE
    "#b07aa1",
    "#76b7b2",
    "#ff9da7",
    "#edc948",
    "#bab0ac",
]


def _extrapolate_basis_set(En, Em, n, exponential):
    assert n in [2, 3, 4]
    m = n + 1
    if exponential:
        alpha = 4.42 if n == 2 else 5.46
        cn, cm = np.exp(n * alpha), np.exp(m * alpha)
    else:
        beta = 2.46 if n == 2 else 3.05
        cn, cm = n**beta, m**beta
    E_cbs = (En * cn - Em * cm) / (cn - cm)
    return E_cbs


def cbs_extrapolate(
    df,
    extrapolate=(2, 3),
    HF_extrapolate=(3, 4),
    HF_method="UHF",
    cbs_name_HF="CBS",
    cbs_name_rest="CBS",
):
    df = df.copy()
    basis_set_sizes = {
        "cc-pVDZ": 2,
        "cc-pVTZ": 3,
        "cc-pVQZ": 4,
        "cc-pV5Z": 5,
        "aug-cc-pVDZ": 2,
        "aug-cc-pVTZ": 3,
        "aug-cc-pVQZ": 4,
        "aug-cc-pV5Z": 5,
        "def2-SVP": 2,
        "def2-TZVP": 3,
        "def2-QZVP": 4,
    }
    df["basis_set"] = df["basis_set"].map(basis_set_sizes)
    df_hf = df[df["method"] == HF_method].copy().rename(columns={"E_final": "E_hf"})
    df = df.merge(df_hf[["comment", "basis_set", "E_hf"]], on=["comment", "basis_set"], how="left")

    pivot = df.pivot_table(
        index=["method", "comment"], columns="basis_set", values=["E_final", "E_hf"], aggfunc="mean"
    ).reset_index()
    pivot_hf = pivot[pivot["method"] == HF_method]
    pivot_rest = pivot[pivot["method"] != HF_method]

    E_hf_cbs = _extrapolate_basis_set(
        pivot_hf["E_final"][HF_extrapolate[0]], pivot_hf["E_final"][HF_extrapolate[1]], HF_extrapolate[0], True
    )
    if isinstance(extrapolate, tuple):
        assert len(extrapolate) == 2 and (extrapolate[0] + 1 == extrapolate[1])
        E1 = pivot_rest["E_final"][extrapolate[0]] - pivot_rest["E_hf"][extrapolate[0]]
        E2 = pivot_rest["E_final"][extrapolate[1]] - pivot_rest["E_hf"][extrapolate[1]]
        E_corr = _extrapolate_basis_set(E1, E2, extrapolate[0], False)
    elif isinstance(extrapolate, int):
        E_corr = pivot_rest["E_final"][extrapolate] - pivot_rest["E_hf"][extrapolate]

    df_hf_cbs = pd.DataFrame(
        {"E_final": E_hf_cbs, "method": HF_method, "basis_set": cbs_name_HF, "comment": pivot_hf.comment}
    )
    df_rest_cbs = pd.DataFrame(
        {"E_corr": E_corr, "method": pivot_rest.method, "basis_set": cbs_name_rest, "comment": pivot_rest.comment}
    )
    df_rest_cbs = df_rest_cbs.merge(df_hf_cbs[["comment", "E_final"]], on="comment", how="left")
    df_rest_cbs["E_final"] += df_rest_cbs["E_corr"]
    df_rest_cbs = df_rest_cbs.drop(columns=["E_corr"])
    return pd.concat([df_hf_cbs, df_rest_cbs], axis=0, ignore_index=True)
