# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib
import colorsys


def format_value_with_error(value, error):
    if np.isnan(value):
        return "--"
    error_rounded = float(np.format_float_positional(error, precision=1, fractional=False, unique=False, trim="k"))
    n_digits = int(np.ceil(-np.log10(error_rounded)))
    assert 0 < error_rounded < 10
    value_rounded = np.round(value, n_digits)
    s = f"{{val:.{n_digits}f}}".format(val=value_rounded)
    s += f"({int(error_rounded * 10**n_digits)})"
    return s


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


def get_outlier_mask(x, window_size=200, quantile=0.1, outlier_range=10):
    n_samples = len(x)
    if not window_size:
        window_size = n_samples
    window_size = min(window_size, n_samples)
    n_blocks = int(np.ceil(n_samples / window_size))
    padding = np.zeros(n_blocks * window_size - n_samples) * np.nan
    x_blocks = np.concat([padding, x]).reshape([n_blocks, window_size])
    quantiles = np.quantile(x_blocks, [quantile, 0.5, 1 - quantile], axis=1)
    spread = quantiles[2] - quantiles[0]
    cutoff_low = quantiles[1] - outlier_range * spread
    cutoff_high = quantiles[1] + outlier_range * spread
    is_outlier = (x_blocks < cutoff_low[:, None]) | (x_blocks > cutoff_high[:, None])
    is_outlier = is_outlier.flatten()[-n_samples:] | (~np.isfinite(x))
    return is_outlier


def fit_with_joint_slope(x_values, E_values):
    n_curves = len(x_values)
    assert len(E_values) == n_curves
    y = np.concat(E_values)
    X = []
    for i in range(n_curves):
        X.append(np.concat([np.ones_like(x) * (i == j) for j, x in enumerate(x_values)]))
    X.append(np.concat(x_values))
    X = np.stack(X, axis=1)
    coeffs = np.linalg.lstsq(X, y)[0]
    return coeffs[:-1], coeffs[-1]


def extrapolate_relative_energy(step, x_values, E_values, method="same_slope", min_frac_step=0.5, return_slopes=False):
    include = step >= (min_frac_step * np.max(step))
    x_values_raw = [x[include] for x in x_values]
    E_values_raw = [E[include] for E in E_values]

    x_values, E_values = [], []
    for x, E in zip(x_values_raw, E_values_raw):
        include = np.isfinite(x) & np.isfinite(E)
        x_values.append(x[include])
        E_values.append(E[include])

    if method == "same_slope":
        E_fit, slope = fit_with_joint_slope(x_values, E_values)
        slopes = np.ones_like(E_fit) * slope
        if slope < 0:
            print("Warning: negative slope, energy etxtrapolation probably unreliable")
    elif method in ["extrapolate", "match", "extrapolate_0"]:
        results = [linregress(x, E) for x, E in zip(x_values, E_values)]
        slopes = np.array([r.slope for r in results])
        intercepts = np.array([r.intercept for r in results])
        if np.any(slopes < 0):
            print("Warning: negative slope, energy etxtrapolation probably unreliable")
        if method == "extrapolate":
            x = min([x[-1] for x in x_values])
        elif method == "extrapolate_0":
            x = 0
        else:
            x = max([x[-1] for x in x_values])
        E_fit = intercepts + x * slopes
    else:
        raise ValueError(f"Unknown extrapolation method: {method}")
    if return_slopes:
        return E_fit, slopes
    return E_fit


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
    elif extrapolate is None:
        E_corr = None

    df_hf_cbs = pd.DataFrame(
        {"E_final": E_hf_cbs, "method": HF_method, "basis_set": cbs_name_HF, "comment": pivot_hf.comment}
    )
    if E_corr is not None:
        df_rest_cbs = pd.DataFrame(
            {"E_corr": E_corr, "method": pivot_rest.method, "basis_set": cbs_name_rest, "comment": pivot_rest.comment}
        )
        df_rest_cbs = df_rest_cbs.merge(df_hf_cbs[["comment", "E_final"]], on="comment", how="left")
        df_rest_cbs["E_final"] += df_rest_cbs["E_corr"]
        df_rest_cbs = df_rest_cbs.drop(columns=["E_corr"])
        df_hf_cbs = pd.concat([df_hf_cbs, df_rest_cbs], axis=0, ignore_index=True)
    return df_hf_cbs


def focal_point_analysis(energies, method_strings):
    methods = [m.split("/") for m in method_strings]
    assert all([len(m) == 2 for m in methods])  # pairs of method, basis set
    E = energies[method_strings[0]]
    for (method, basis_set), (larger_meth, _) in methods[1:], methods[:-1]:
        E += energies[method + "/" + basis_set] - energies[larger_meth + "/" + basis_set]
    return E


def scale_lightness(color, scale_l):
    rgb = matplotlib.colors.ColorConverter.to_rgb(color)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)  # noqa
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


MILLIHARTREE = "[m$E_\\text{h}$]"

if __name__ == "__main__":
    x = np.linspace(0, 1, 10_000)
    x[70] = 100

    mask = get_outlier_mask(x)
    print(np.sum(mask))
