#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fname = "timings_batchsize_32.txt"
data = []
with open(fname, "r") as f:
    for line in f:
        line = line.strip().replace("Summary: ", "").replace("sparse time", "t")
        data.append(eval("dict(" + line + ")"))
df = pd.DataFrame(data)


plt.close("all")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=df, x="n_el", y="t", hue="cutoff", ax=ax)

