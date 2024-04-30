# %%
import argparse
import itertools
import yaml
import pathlib
import copy
import os
import subprocess
import shutil
import random

SPECIAL_KEYS = ["slurm"]
N_PARAM_GROUPS = 5


def update_dict(original, update, allow_new_keys):
    original = copy.deepcopy(original)
    for key, value in update.items():
        if isinstance(value, dict):
            if (key not in original) and (not allow_new_keys):
                raise KeyError(f"Key {key} not found in original dict. Possible keys are {list(original.keys())}")
            original_subdict = original.get(key, {})
            original[key] = update_dict(original_subdict, update[key], allow_new_keys)
        else:
            original[key] = value
    return original


def convert_to_default_datatype(config_dict, default_dict):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            config_dict[key] = convert_to_default_datatype(config_dict[key], default_dict[key])
        else:
            config_dict[key] = type(default_dict[key])(config_dict[key])
    return config_dict


def load_yaml(fname):
    with open(fname, "r") as f:
        return yaml.load(f, Loader=yaml.CLoader)


def save_yaml(fname, data):
    with open(fname, "w") as f:
        yaml.dump(data, f, Dumper=yaml.CDumper)


def to_nested_dict(flattened_dict):
    nested_dict = {}
    for key, value in flattened_dict.items():
        keys = key.split(".")
        current_dict = nested_dict
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return nested_dict


def setup_run_dir(run_dir, run_config, full_config, force):
    if os.path.exists(run_dir):
        if force:
            print(f"Directory {run_dir} already exists, overwriting.")
            shutil.rmtree(run_dir)
        else:
            print(f"Skipping existing run {run_dir}")
            return False
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(os.path.join(run_dir, "config.yaml"), run_config)
    save_yaml(os.path.join(run_dir, "full_config.yaml"), full_config)
    return True


def get_slurm_template(cluster=None):
    if cluster is None:
        cluster = "hgx"
    template_fname = pathlib.Path(__file__).parent / "../../config/slurm_templates" / f"{cluster}.sh"
    with open(template_fname, "r") as f:
        return f.read()


def submit_to_slurm(run_dir, slurm_config, dry_run=False):
    current_dir = os.getcwd()
    os.chdir(run_dir)

    cluster = autodetect_cluster()
    slurm_template = get_slurm_template(cluster)
    slurm_defaults = get_slurm_defaults(cluster, slurm_config.get("queue"))
    slurm_config = slurm_defaults | slurm_config
    job_file = slurm_template.format(**slurm_config)
    with open("job.sh", "w") as f:
        f.write(job_file)

    if not dry_run:
        subprocess.run(["sbatch", "job.sh"])
    os.chdir(current_dir)


def get_slurm_defaults(cluster, queue):
    if cluster == "hgx":
        return dict(time="30-00:00:00", n_gpus=1)
    elif cluster == "vsc5":
        defaults = dict(time="3-00:00:00", n_gpus=2)
        if queue == "a100":
            defaults["partition"] = "zen3_0512_a100x2"
            defaults["qos"] = "zen3_0512_a100x2"
        elif queue == "a40":
            defaults["partition"] = "zen2_0256_a40x2"
            defaults["qos"] = "zen2_0256_a40x2"
        return defaults


def autodetect_cluster():
    hostname = os.uname()[1]
    if hostname == "gpu1-mat":
        return "hgx"
    elif "vsc" in hostname:
        return "vsc5"


def build_grid(*param_groups):
    all_keys = []
    group_values = []
    for param_group in param_groups:
        all_keys += [p[0] for p in param_group]
        values = [p[1:] for p in param_group]
        values = list(zip(*values))
        group_values.append(values)

    grid = []
    for values in itertools.product(*group_values):
        all_values = [v for value in values for v in value]
        grid.append({k: v for k, v in zip(all_keys, all_values)})
    return grid


def setup_calculations():
    parser = argparse.ArgumentParser(
        description="Setup and dispatch one or multiple calculations, e.g. for a parameter sweep"
    )
    parser.add_argument("--input", "-i", default="config.yaml", help="Path to input config file")
    parser.add_argument("--parameter", "-p", nargs="+", action="append", default=[])
    for n in range(N_PARAM_GROUPS):
        parser.add_argument(f"--parameter{n}", f"-p{n}", nargs="+", action="append", default=[])
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite directories if they already exist")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only set-up the directories and config-files, but do not dispatch the actual calculation.",
    )
    # args = parser.parse_args("--force --dry-run -p1 pretraining.steps 1 2 3 -p1 optimization.steps 1000 2000 3000 -p experiment_name e1 e2 -p seed 10 20".split())
    args = parser.parse_args()

    # Load the default config and the local config file
    default_path = pathlib.Path(__file__).parent / "../../config/default.yaml"
    default_config = load_yaml(default_path)
    file_config = load_yaml(args.input)

    # Merge the default config with the input config file; we'll not actually use this merge,
    # but this way we can report errors in the config file
    try:
        update_dict(
            default_config, {k: v for k, v in file_config.items() if k not in SPECIAL_KEYS}, allow_new_keys=False
        )
    except KeyError as e:
        print("Error merging default config with input config file.")
        raise e

    # Split cli overrides into param names (=keys) and values
    param_groups = [getattr(args, f"parameter{n}") for n in range(N_PARAM_GROUPS)]
    param_groups = [p for p in param_groups if p]
    for p in args.parameter:
        param_groups.append([p])
    cli_override_grid = build_grid(*param_groups)

    # Loop over grid of all value combinations
    for cli_dict in cli_override_grid:
        cli_values = list(cli_dict.values())
        # Merge file config with cli dict.
        cli_dict = to_nested_dict(cli_dict)
        run_config = update_dict(file_config, cli_dict, allow_new_keys=True)

        # Pop special keys, which are purely used for setup
        slurm_config = run_config.pop("slurm", {})

        # Merge cli dict into base config (which is default + file config)
        try:
            full_config = update_dict(default_config, run_config, allow_new_keys=False)
        except KeyError as e:
            print("Error merging default config with cli overrides.")
            raise e
        run_config = convert_to_default_datatype(run_config, default_config)
        full_config = convert_to_default_datatype(full_config, default_config)

        # Generate the name for the run
        run_name = "_".join([str(v) for v in cli_values])
        if full_config["experiment_name"]:
            run_name = f'{full_config["experiment_name"]}_{run_name}'

        # Pick a random seed for this run, unless specified
        if full_config.get("seed", -1) < 0:
            full_config["seed"] = random.randint(0, 2**16 - 1)

        # Create the run-directory and submit the job with slurm
        success = setup_run_dir(run_name, run_config, full_config, args.force)
        if success:
            slurm_config["job_name"] = run_name
            submit_to_slurm(run_name, slurm_config, args.dry_run)


if __name__ == "__main__":
    setup_calculations()
