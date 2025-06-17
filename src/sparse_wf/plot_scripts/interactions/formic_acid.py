# %%
import pandas as pd
from sparse_wf.plot_utils import cbs_extrapolate, focal_point_analysis

molecule = "Formic acid dimer"
# molecule = "Water dimer"


def rename_basis_set(b):
    if b.startswith("CBS"):
        return b
    if b.endswith("DZ"):
        return "2"
    if b.endswith("TZ"):
        return "3"
    if b.endswith("QZ"):
        return "4"
    if b.endswith("5Z"):
        return "5"
    raise ValueError(f"Unknown basis set {b}")


df = pd.read_csv("orca_energies.csv")
# df = df[df.method.isin(["HF", "MP2"])]
df = df[df.basis_set.str.startswith("aug") & (~df.method.str.contains("UHF"))]
# df = df[~(df.method.isin(["HF", "MP2"]) & df.basis_set.str.startswith("cc-pV"))]
df_hfcbs23 = cbs_extrapolate(df, None, (2, 3), "HF", "CBS23", None)
df_hfcbs34 = cbs_extrapolate(df, None, (3, 4), "HF", "CBS34", None)
df_23 = cbs_extrapolate(df, (2, 3), (4, 5), "HF", "CBS45", "CBS23")
df_34 = cbs_extrapolate(df, (3, 4), (4, 5), "HF", "CBS45", "CBS34")
df_45 = cbs_extrapolate(df, (4, 5), (4, 5), "HF", "CBS45", "CBS45")
df = pd.concat([df, df_hfcbs23, df_hfcbs34, df_23, df_34, df_45], axis=0, ignore_index=True)
df["basis_set"] = df["basis_set"].apply(rename_basis_set)
df["method"] = df["method"] + "/" + df["basis_set"]
# df = df[df.method.str.contains("CCSD(T)/CBS", regex=False) | df.method.str.contains("CCSDT")]

df["molecule"] = df["comment"].apply(lambda x: " ".join(x.split("_")[1:]).replace(" Dissociated", ""))
df["geom"] = df["comment"].apply(lambda c: "dissociated" if "Dissociated" in c else "equilibrium")

pivot = df.pivot_table(index=["molecule", "method"], columns="geom", values="E_final")
pivot["deltaE"] = (pivot["dissociated"] - pivot["equilibrium"]) * 1000
pivot = pivot.reset_index()
pivot = pivot[pivot.molecule == molecule].sort_values("method").set_index("method")
E = pivot["deltaE"]
print(pivot)


fpa_methods = [
    ["MP2/CBS34", "CCSD(T)/3"],
    ["HF/CBS45", "MP2/CBS34", "CCSD(T)/3"],
]
print("\nFocal point analysis:")
for m in fpa_methods:
    method_str = " -> ".join(m)
    print(f"{method_str:<40}: {focal_point_analysis(E, m):.2f} mHa")
