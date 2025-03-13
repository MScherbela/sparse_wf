import argparse
import os
import shutil
import subprocess
import time
import pathlib
import json
import itertools
import multiprocessing

ANGSTROM_IN_BOHR = 1.8897259886
PERIODIC_TABLE = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()


def write_orca_input(R, Z, charge, spin, fname, method, basis_set, no_frozen_core, n_proc, memory_per_core):
    multiplicity = int(2*spin+1)
    if "+" in method:
        scf_method, method = method.split("+")
    else:
        scf_method = None
    command = f"!{method}"
    if no_frozen_core:
        command += " NoFrozenCore"
    else:
        command += " FrozenCore"
    command += f" {basis_set}"
    command += " SlowConv"

    if (("DLPNO" in method) or ("RI-MP2" in method)) and not ("F12" in method):
        command += f" {basis_set}/C" # auxiliary basis
    if "F12" in method:
        assert basis_set in ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "cc-pV5Z"]
        # Switch to F12 basis and add full cabs basis set
        command += f"-F12 aug-{basis_set}/C {basis_set}-F12-CABS"
    # if ("CCSD" in method) and spin:
    #     command += " UNO"
    with open(fname, "w") as f:
        f.write(f"{command}\n")
        fname = os.path.abspath(fname)
        if scf_method:
            mo_fname = fname.replace("orca.inp", "orca.gbw").replace(f"{scf_method}+{method}", scf_method)
            f.write("!moread\n")
            f.write(f'%moinp "{mo_fname}"\n')
            f.write("!NoIter\n")
        else:
            f.write("%SCF\n")
            f.write("  Convergence VeryTight\n")
            f.write("  maxiter 300\n")
            f.write("  ConvForced 1\n")
            f.write("END\n")
        if "CCSD" in method and "CCSDT" not in method:
            f.write("%MDCI\n")
            f.write("  CIType CCSD\n")
            if "DLPNO" in method:
                f.write("  UseFullLMP2Guess false\n") # better comparison between open shell/closed shell calcs
            f.write("  NatOrbs true\n")
            f.write("  maxiter 50\n")
            f.write("  DIISStartIter 0")
            f.write("  MaxDIIS 25\n")
            f.write("  LShift 0.5\n")
            # f.write("  UseQROs true\n")
            # f.write("  PrintLevel 4\n")
            f.write("END\n")
        if "CCSDT" in method:
            f.write("%AUTOCI\n")
            f.write("  citype CCSDT\n")
            f.write("END\n")
        if ("CAS" in method) or ("NEVPT2" in method):
            # %casscf nel  4  # number of active space electrons
            f.write("%CASSCF\n")
            f.write("  nel 4\n")
            f.write("  norb 4\n")
            f.write(f"  mult {multiplicity}\n")
            if "CASCI" in method:
                f.write("  maxiter 1\n")
            else:
                f.write("  maxiter 200\n")
            f.write("  orbstep SUPERCI\n")
            f.write("  switchstep DIIS\n")
            f.write("  MaxDIIS 20\n")
            f.write("  DIISThresh 1e-7\n")
            # f.write("  ShiftUp 2.0\n")
            # f.write("  ShiftDn 2.0\n")
            # f.write("  MinShift 0.6\n")
            f.write("  maxrot 0.15\n")
            f.write("END\n")
        if "MP2" in method:
            f.write("%MP2\n")
            f.write("  NatOrbs true\n")
            f.write("END\n")
        f.write(f"%MAXCORE {memory_per_core}\n")
        f.write("%PAL\n")
        f.write(f"  nprocs {n_proc}\n")
        f.write("END\n")
        f.write(f"* xyz {charge} {multiplicity}\n")
        for r, z in zip(R, Z):
            f.write(
                f" {PERIODIC_TABLE[z-1]} {r[0] / ANGSTROM_IN_BOHR} {r[1] / ANGSTROM_IN_BOHR} {r[2] / ANGSTROM_IN_BOHR}\n"
            )
        f.write("*\n")


def get_orca_results(output_fname):
    results = {}
    with open(output_fname, "r") as f:
        for line in f:
            if ("Total Energy       :" in line) and ("E_hf" not in results):
                results["E_hf"] = float(line.split()[3])
            if ("FINAL SINGLE POINT ENERGY" in line) and ("E_final" not in results):
                results["E_final"] = float(line.split()[4])
    return results


def write_input_files(g, directory, method, basis_set, no_frozen_core, n_proc, total_memory):
    write_orca_input(
        g["R"],
        g["Z"],
        g.get("charge", 0),
        g.get("spin", 0) / 2,
        os.path.join(directory, "orca.inp"),
        method,
        basis_set,
        no_frozen_core,
        n_proc,
        int(total_memory * 1000 / n_proc),
    )
    with open(os.path.join(directory, "metadata.json"), "w") as f:
        json.dump(dict(
            method=method,
            basis_set=basis_set,
            frozen_core=not no_frozen_core,
            n_proc=n_proc,
            total_memory=total_memory,
            geom_name=g.get("comment", ""),
            geom_hash=g.get("hash", ""),
            geom_R=g.get("R", []),
            geom_Z=g.get("Z", []),
        ), f, indent=4)

def run_orca(g, directory, method, basis_set, no_frozen_core, n_proc, total_memory, orca_path="orca", clean_calc_dir=True):
    write_input_files(g, directory, method, basis_set, no_frozen_core, n_proc, total_memory)
    with open(os.path.join(directory, "orca.out"), "w") as f:
        subprocess.call([orca_path, "orca.inp"], cwd=directory, stdout=f, stderr=f)
    results = get_orca_results(os.path.join(directory, "orca.out"))
    if clean_calc_dir:
        for fname in os.listdir(directory):
            if fname not in ["orca.out", "orca.inp"]:
                os.remove(os.path.join(directory, fname))
    return results


def worker(args):
    ind_calc, geom_hash, g, method, basis_set, n_proc, total_memory, orca_path, no_frozen_core = args
    g["hash"] = geom_hash
    geom_comment = g.get("comment", "")
    directory = f"{geom_comment}_{method}_{basis_set}"
    directory = directory.replace("/", "_")
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    t0 = time.time()
    results = run_orca(g, directory, method, basis_set, no_frozen_core, n_proc, total_memory, orca_path)
    t1 = time.time()
    return (ind_calc, geom_hash, geom_comment, method, basis_set, results, t1 - t0)

def submit_to_slurm(args):
    ind_calc, geom_hash, g, method, basis_set, comment, n_proc, total_memory, orca_path, no_frozen_core = args
    g["hash"] = geom_hash
    geom_comment = g.get("comment", "")
    directory = f"{geom_comment}_{method}_{basis_set}"
    if comment:
        directory += f"_{comment}"
    directory = directory.replace("/", "_")
    if os.path.isdir(directory):
        print("Skipping existing directory", directory)
        return
    os.makedirs(directory)
    write_input_files(g, directory, method, basis_set, no_frozen_core, n_proc, total_memory)
    job_name = f"orca_{geom_comment}_{method}_{basis_set}".replace(" ", "_")
    with open(f"{directory}/job.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output=orca.out\n")
        f.write(f"#SBATCH --time=3-00:00:00\n")
        f.write(f"#SBATCH --ntasks={n_proc}\n")
        f.write(f"#SBATCH --mem={total_memory}G\n")
        f.write(f"#SBATCH --partition=hgx\n")
        f.write(f"#SBATCH --qos=cpusonly\n")
        f.write(f"#SBATCH --cpus-per-task=1\n")
        f.write(f"{orca_path} orca.inp\n")
    subprocess.call(["sbatch", "job.sh"], cwd=directory)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--method", type=str, nargs="+", required=False, default="CCSD(T)")
    parser.add_argument("--basis-set", type=str, nargs="+", required=False, default="aug-cc-pVDZ")
    parser.add_argument("--geometry-hashes", type=str, nargs="+", default=None, required=False)
    parser.add_argument("--geometry-comments", type=str, nargs="+", default=None, required=False)
    parser.add_argument("--n-proc", type=int, default=16, required=False)
    parser.add_argument("--total-memory", type=int, default=200, required=False, help="Total memory in GB")
    parser.add_argument("--n-parallel", type=int, default=1, required=False, help="Number of parallel workers")
    parser.add_argument("--slurm", action="store_true", required=False, help="Submit to SLURM")
    parser.add_argument("--comment", type=str, nargs="+", default=[""], required=False, help="Comment for the calculation")

    parser.add_argument(
        "--orca-path",
        type=str,
        default="/gpfs/data/fs71573/scherbela/orca/orca",
        required=False,
        help="Full path to ORCA executable",
    )
    parser.add_argument("--no-frozen-core", action="store_true", required=False, help="Do not use frozen core approximation")
    args = parser.parse_args()

    db_fname = pathlib.Path(__file__).parent / "../../../../data/geometries.json"
    with open(db_fname, "r") as f:
        all_geometries = json.load(f)

    geometry_hashes = args.geometry_hashes or []
    geometry_comments = args.geometry_comments or []
    for comment in geometry_comments:
        geoms = [k for k, v in all_geometries.items() if v.get("comment", "") == comment]
        assert len(geoms) == 1, f"Comment not unique: {comment}"
        geometry_hashes.append(geoms[0])

    constant_args = (
        args.n_proc,
        args.total_memory,
        args.orca_path,
        args.no_frozen_core,
    )
    calc_args = []
    for ind_calc, (method, basis_set, geom_hash, comment) in enumerate(
        itertools.product(
            args.method,
            args.basis_set,
            geometry_hashes,
            args.comment
        )
    ):
        calc_args.append((ind_calc, geom_hash, all_geometries[geom_hash], method, basis_set, comment, *constant_args))
    if args.slurm:
        for arg in calc_args:
            submit_to_slurm(arg)
    else:
        with open("energies.csv", "w", buffering=1) as energy_file:
            energy_file.write("ind_calc;geom_hash;comment;method;basis_set;E_hf;E_final;duration\n")
            with multiprocessing.Pool(processes=args.n_parallel) as pool:
                results = pool.imap_unordered(worker, calc_args)
                for ind_calc, geom_hash, geom_comment, method, basis_set, result, duration in results:
                    result_str = f"{ind_calc};{geom_hash};{geom_comment};{method};{basis_set};{result.get('E_hf', '')};{result.get('E_final', '')};{duration}"
                    energy_file.write(result_str + "\n")
                    print(result_str)
