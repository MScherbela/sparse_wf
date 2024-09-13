#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import itertools
import yaml
import pathlib
import copy
import os
import subprocess
import shutil
import random
import argcomplete
from argcomplete.completers import ChoicesCompleter, FilesCompleter

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent / "../../config/default.yaml"
SPECIAL_KEYS = ["slurm"]
N_PARAM_GROUPS = 5


def get_all_parameter_keys():
    default_path = pathlib.Path(__file__).parent / "../../config/default.yaml"
    default_config = load_yaml(default_path)
    default_config = to_flat_dict(default_config)
    return list(default_config.keys())


def update_dict(original, update, allow_new_keys):
    original = copy.deepcopy(original)
    for key, value in update.items():
        if isinstance(value, dict):
            if isinstance(original, dict):
                if (key not in original) and (not allow_new_keys):
                    raise KeyError(f"Key {key} not found in original dict. Possible keys are {list(original.keys())}")
                original_subdict = original.get(key, {})
                original[key] = update_dict(original_subdict, update[key], allow_new_keys)
            elif isinstance(original, list):
                list_index = int(key)
                if list_index >= len(original):
                    raise KeyError(
                        f"List index {list_index} exceeds length of list. Maximum index is {len(original) - 1}."
                    )
                original[list_index] = update_dict(original[list_index], update[key], allow_new_keys)
        else:
            if (key not in original) and (not allow_new_keys):
                raise KeyError(f"Key {key} not found in original dict. Possible keys are {list(original.keys())}")
            original[key] = value
    return original


def convert_to_default_datatype(config_dict, default_dict, allow_new_keys=False):
    if isinstance(config_dict, dict):
        for key, value in config_dict.items():
            if key not in default_dict:
                if allow_new_keys:
                    continue
                else:
                    raise KeyError(f"Key {key} not found in default dict.")
            config_dict[key] = convert_to_default_datatype(value, default_dict[key], allow_new_keys)
    elif isinstance(config_dict, list):
        for i, value in enumerate(config_dict):
            if len(default_dict) != 0:
                config_dict[i] = convert_to_default_datatype(value, default_dict[0], allow_new_keys=True)
    elif config_dict is None:
        pass
    else:
        target_type = type(default_dict)
        if (target_type is bool) and isinstance(config_dict, str):
            assert config_dict.lower() in ["true", "false", "0", "1"]
            config_dict = config_dict.lower() in ["true", "1"]
        else:
            config_dict = target_type(config_dict)
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


def to_flat_dict(nested_dict, parent_key=""):
    flat_dict = {}
    for key, value in nested_dict.items():
        if parent_key:
            key = f"{parent_key}.{key}"
        if isinstance(value, dict):
            flat_dict.update(to_flat_dict(value, key))
        else:
            flat_dict[key] = value
    return flat_dict


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
    if "n_gpus" in slurm_config:
        slurm_config["n_gpus"] = int(slurm_config["n_gpus"])
    job_file = eval('f"""' + slurm_template + '"""', None, slurm_config)
    with open("job.sh", "w") as f:
        f.write(job_file)

    if not dry_run:
        subprocess.run(["sbatch", "job.sh"])
    os.chdir(current_dir)


def get_slurm_defaults(cluster, queue):
    if cluster == "hgx":
        defaults = dict(time="30-00:00:00", n_gpus=1, qos="normal")
    elif cluster == "vsc5":
        defaults = dict(time="3-00:00:00", n_gpus=2)
        if queue == "a100":
            defaults["partition"] = "zen3_0512_a100x2"
            defaults["qos"] = "zen3_0512_a100x2"
        elif queue == "a40":
            defaults["partition"] = "zen2_0256_a40x2"
            defaults["qos"] = "zen2_0256_a40x2"
    elif cluster == "leonardo":
        defaults = dict(time="1-00:00:00", n_gpus=4)
    return defaults


def autodetect_cluster():
    hostname = os.uname()[1]
    if hostname == "gpu1-mat":
        return "hgx"
    elif "vsc" in hostname:
        return "vsc5"
    elif "leonardo" in hostname:
        return "leonardo"


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


def get_argparser():
    param_choices = get_all_parameter_keys()
    param_completer = ChoicesCompleter(param_choices)
    yaml_completer = FilesCompleter(["yaml", "yml"])

    parser = argparse.ArgumentParser(
        description="Setup and dispatch one or multiple calculations, e.g. for a parameter sweep"
    )
    parser.add_argument(
        "--input", "-i", default="config.yaml", help="Path to input config file"
    ).completer = yaml_completer  # type: ignore
    parser.add_argument("--parameter", "-p", nargs="+", action="append", default=[]).completer = param_completer  # type: ignore
    for n in range(N_PARAM_GROUPS):
        parser.add_argument(
            f"--parameter{n}", f"-p{n}", nargs="+", action="append", default=[]
        ).completer = param_completer  # type: ignore
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite directories if they already exist")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only set-up the directories and config-files, but do not dispatch the actual calculation.",
    )
    parser.add_argument("--no-commit-check", action="store_true", help="Do not check if code is committed.")
    argcomplete.autocomplete(parser)
    return parser


def is_code_committed():
    code_path = pathlib.Path(__file__).parent.parent
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=code_path, capture_output=True)
    return not git_status.stdout


def setup_calculations():
    parser = get_argparser()
    args = parser.parse_args()

    if (not args.no_commit_check) and (not is_code_committed()):
        print(
            "Warning: Code has uncommited changes. Running calculations might not be reproducible. Proceed anyway [y/N]?"
        )
        if input().lower() != "y":
            return

    # Load the default config and the local config file
    default_config = load_yaml(DEFAULT_CONFIG_PATH)
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
        experiment_name = full_config.get("metadata", {}).get("experiment_name", "exp")
        run_name = "_".join([str(v) for v in cli_values])
        if run_name:
            run_name = f"{experiment_name}_{run_name}"
        else:
            run_name = experiment_name

        # Pick a random seed for this run, unless specified
        if full_config.get("seed", 0) == 0:
            full_config["seed"] = random.randint(0, 2**16 - 1)

        # Create the run-directory and submit the job with slurm
        success = setup_run_dir(run_name, run_config, full_config, args.force)
        if success:
            slurm_config["job_name"] = run_name
            submit_to_slurm(run_name, slurm_config, args.dry_run)


if __name__ == "__main__":
    setup_calculations()
