#%%
import pandas as pd
from sparse_wf.plot_utils import cbs_extrapolate

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

df = pd.read_csv('orca_energies.csv')
# df = df[df.method.isin(["HF", "MP2"])]
df = df[df.basis_set.str.startswith("aug")]
# df = df[~(df.method.isin(["HF", "MP2"]) & df.basis_set.str.startswith("cc-pV"))]
df_23 = cbs_extrapolate(df, (2, 3), (4,5), "HF", "CBS45", "CBS23")
df_34 = cbs_extrapolate(df, (3, 4), (4,5), "HF", "CBS45", "CBS34")
df_45 = cbs_extrapolate(df, (4, 5), (4,5), "HF", "CBS45", "CBS45")
df = pd.concat([df, df_23, df_34, df_45], axis=0, ignore_index=True)
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

def get_FPA(E, method_fast, method_slow, basis_small, basis_large):
    return E[method_fast + "/" + basis_large] + E[method_slow + "/" + basis_small] - E[method_fast + "/" + basis_small]

# # print(E["MP2/CBS_45"] - E["MP2/aug-cc-pV5Z"])

# print("FPA for CCSD(T)")
# print(get_FPA(E, "MP2", "CCSD(T)", "4", "5"))
# print(get_FPA(E, "MP2", "CCSD(T)", "4", "CBS45"))
# print(get_FPA(E, "MP2", "CCSD(T)", "CBS34", "CBS45"))


# # print(get_FPA(E, "MP2", "CCSD(T)", "cc-pVQZ", "cc-pV5Z"))
# # print(get_FPA(E, "MP2", "CCSD(T)", "CBS_34", "CBS_45"))

# print("FPA for CCSDT")
# # print(get_FPA(E, "MP2", "CCSDT", "2", "CBS45"))
# print(get_FPA(E, "MP2", "CCSDT", "3", "CBS45"))
# print(get_FPA(E, "MP2", "CCSDT", "CBS23", "CBS45"))

# E_fpa = E["MP2/CBS45"] + (E["CCSD(T)/4"] - E["MP2/4"]) + (E["CCSDT/3"] - E["CCSD(T)/3"])
# print(E_fpa)


# print("FPA for NEVPT2")
# print(get_FPA(E, "MP2", "NEVPT2", "CBS_23", "CBS_45"))
