from sparse_wf.geometry import load_geometries, PERIODIC_TABLE, BOHR_IN_ANGSTROM
import os
import shutil
import subprocess
import numpy as np

def get_orca_geom_string(R, Z):
    s = ""
    for r, z in zip(R, Z):
        s += f" {PERIODIC_TABLE[z-1]} {r[0] * BOHR_IN_ANGSTROM} {r[1] * BOHR_IN_ANGSTROM} {r[2] * BOHR_IN_ANGSTROM}\n"
    return s[:-1]

def get_constraint_string(n_carbon):
    return f"{{D {n_carbon} 0 {n_carbon-1} {n_carbon+2} C}}\n{{D {n_carbon} 0 {n_carbon-1} {n_carbon+3} C}}\n{{D {n_carbon+1} 0 {n_carbon-1} {n_carbon+2} C}}\n{{D {n_carbon+1} 0 {n_carbon-1} {n_carbon+3} C}}"


all_geoms = list(load_geometries().values())

orca_template = open("template.inp").read()
for n_carbon in [20, 30]:
    for angle in [0, 90]:
        for spin_state in ["singlet", "triplet"]:
            if angle == 0 and spin_state == "triplet":
                continue
            geom_name = f"cumulene_C{n_carbon}H4_{angle}deg_{spin_state}"
            geom = [g for g in all_geoms if g.comment == geom_name][0]
            if os.path.exists(geom_name):
                print("Skipping", geom_name)
                continue
            os.makedirs(geom_name)
            os.chdir(geom_name)

            # Bias towards dimerization
            R = geom.R.copy()
            R *= 0.95
            R[1:-4:2, 0] += 0.01
            R[2:-4:2, 0] -= 0.01
            R[:-4, 0] += np.random.normal(0, 0.01, n_carbon)
            orca_inp = orca_template
            orca_inp = orca_inp.replace("GEOM_PLACEHOLDER", get_orca_geom_string(R, geom.Z))
            orca_inp = orca_inp.replace("CONSTRAINT_PLACEHOLDER", get_constraint_string(n_carbon))
            orca_inp = orca_inp.replace("MULT_PLACEHOLDER", str(geom.spin + 1))
            orca_inp = orca_inp.replace("UKS_PLACEHOLDER", "" if geom.spin == 0 else "UKS")

            with open("orca.inp", "w") as f:
                f.write(orca_inp)
            shutil.copy("../job.sh", ".")
            subprocess.call(["sbatch", "-J", geom_name, "job.sh"])
            os.chdir("..")
