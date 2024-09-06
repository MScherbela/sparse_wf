#%%
import numpy as np
from ast import literal_eval
import pandas as pd

fname = "/storage/scherbelam20/runs/sparse_wf/overlaps/overlap_2000/log.txt"
data = []
with open(fname, "r") as f:
    for line in f:
        if "eval/" in line:
            eval_data = literal_eval(line.strip())
            eval_data = {k: v for k, v in eval_data.items() if "eval/overlap_" in k}
            data.append(eval_data)
df = pd.DataFrame(data)
df_summary = df.agg(["mean", "std"]).transpose()
df_summary = df_summary / df_summary["mean"].max()
print(df_summary)