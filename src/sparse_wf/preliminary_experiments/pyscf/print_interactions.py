#%%
import pandas as pd

df = pd.read_csv("/storage/scherbelam20/runs/sparse_wf/hf/energies.csv", header=None)
df.columns = ["molecule", "energy", "s2", "mult"]
df["geom"] = df.molecule.apply(lambda x: "diss" if "Dissociated" in x else "equ")
df["molecule"] = df["molecule"].str.replace("_Dissociated", "")
pivot = df.pivot(index="molecule", columns="geom", values="energy")
pivot["delta"] = pivot["diss"] - pivot["equ"]
print(pivot)