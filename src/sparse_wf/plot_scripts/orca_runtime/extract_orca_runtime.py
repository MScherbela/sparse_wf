# %%
import re
from pathlib import Path
import pandas as pd


def convert_to_total_hours(runtime_str):
    """Converts '0 days 0 hours 0 minutes 13 seconds 687 msec' to decimal hours."""
    # Find all integers in the string: [days, hours, minutes, seconds, msec]
    matches = list(map(int, re.findall(r"(\d+)", runtime_str)))

    if len(matches) == 5:
        days, hours, minutes, seconds, msec = matches
        total_hours = (days * 24) + (hours) + (minutes / 60) + (seconds / 3600) + (msec / 3600000)
        return total_hours
    return None


def parse_orca_output(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    num_atoms_match = re.search(r"Number of atoms\s+...\s+(\d+)", content)
    num_atoms = int(num_atoms_match.group(1))

    basis_match = re.search(r"Your calculation utilizes the basis:\s+(\S+)", content)
    basis_set = basis_match.group(1) if basis_match else "Not found"

    functions_match = re.search(r"Number of basis functions\s+\.\.\.\s+(\d+)", content)
    basis_functions = functions_match.group(1) if functions_match else "Not found"

    success = "****ORCA TERMINATED NORMALLY****" in content

    runtime_match = re.search(r"TOTAL RUN TIME:\s+(.*)", content)
    runtime_str = runtime_match.group(1).strip() if runtime_match else "Not found"
    total_hours = convert_to_total_hours(runtime_str) if runtime_str != "Not found" else "N/A"

    return {
        "num_atoms": num_atoms,
        "basis_set": basis_set,
        "num_basis_functions": basis_functions,
        "success": success,
        "total_hours": total_hours,
    }


all_fnames = Path("/storage/scherbelam20/runs/orca/cumulene/CCSDT_runtime/").glob("*/orca.out")
all_data = []
for fname in all_fnames:
    all_data.append(parse_orca_output(fname))
df = pd.DataFrame(all_data)
df.to_csv("cumulene_orca.csv", index=False)
# %%
