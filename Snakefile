import json

from rules.scripts.custom_types import *

configfile: "config.yaml"

# Define all output directories and prefixes
outdir = pathlib.Path(config["outdir"])
outdir.mkdir(exist_ok=True, parents=True)

dataset_dir = outdir / "{msa}_{model}"
command_dir = dataset_dir /  "{cmd_repr}"
raxmlng_cmd_prefix = command_dir / "{raxmlng}" / "{raxmlng}"
iqtree_cmd_prefix = command_dir / "{raxmlng}" / "{raxmlng}_iqtree"

# Load the run configurations:
# - RAxML-NG versions
# - Datasets and respective models
# - Command lines to run

raxmlng_versions = []
for entry in config["executables"]:
    if len(entry) == 2:
        name, path = entry
        raxmlng_versions.append((name, (path, "")))
    elif len(entry) == 3:
        name, path, extra = entry
        raxmlng_versions.append((name, (path, extra)))
    else:
        raise ValueError(f"Set either two or three values for the RAxML-NG executables in config.yaml. Instead got {len(entry)} values.")

raxmlng_versions = dict(raxmlng_versions)

datasets = config["datasets"]
_msas, _models = zip(*datasets)

msas = dict([(pathlib.Path(msa).name, msa) for msa in _msas])
models = dict([(msa, model) for msa, model in zip(_msas, _models)])
model_names = dict([(msa, model if not pathlib.Path(msa).is_file() else pathlib.Path(model).name) for msa, model in zip(_msas, _models)])



cmds = config["commandLines"]
command_repr_mapping = dict([(cmd
                              .replace(" ", "")
                              .replace("-", "")
                              .replace("{", "")
                              .replace("}", "")
                              , cmd) for cmd in cmds])
with open(outdir / "cmd_mapping.json", "w") as f:
    json.dump(command_repr_mapping,f)


def expand_path(dir: FilePath, expand_command: bool = True, expand_dataset: bool = True):
    
    files = expand(dir,
                  raxmlng=raxmlng_versions,
                  allow_missing=True)

    if expand_dataset:
      files = expand(files,
            zip,
            msa=msas.keys(),
            model=model_names.values(),
            allow_missing=True)

    if expand_command:
      files = expand(files, cmd_repr=command_repr_mapping,)

    return files


def expand_for_suffix(suffix: str, expand_command: bool = True):
    return expand_path(raxmlng_cmd_prefix.with_suffix(suffix), expand_command=expand_command)


rule all:
    input:
        #---------------------
        # Independent RAxML-NG runs and subsequent analyses
        #---------------------
        # RAxML-NG result files
        logfiles = expand_for_suffix(".raxml.log", expand_command=True),
        bestTrees = expand_for_suffix(".raxml.bestTree", expand_command=True),
        mlTrees = expand_for_suffix(".raxml.mlTrees", expand_command=True),

        # IQ-Tree significance analyses
        indiv_iqtree_results = expand_path(iqtree_cmd_prefix.with_suffix(".iqtree"), expand_command=True),

        # collected results
        general_results = expand_for_suffix(".results.parquet", expand_command=True),
        tree_results = expand_for_suffix(".results.trees.parquet", expand_command=True),

        #---------------------
        # Comparison of different RAxML-NG versions
        #---------------------
        # IQ-Tree significance analyses
        collected_trees = expand_path(command_dir / "all.mlTrees", expand_command=True),
        collected_best_tree = expand_path(command_dir / "all.bestTree", expand_command=True),
        versions_file = expand_path(command_dir / "all.versions", expand_command=True),
        consel_file = expand_path(command_dir / "all.consel", expand_command=True),
        sitelh_file = expand_path(command_dir / "all.raxml.siteLH", expand_command=True),
        collected_consel_results = expand_path(command_dir / "collected.results.parquet", expand_command=True),
        all_results = expand_path(command_dir / "all.results.parquet", expand_command=True)

        # collected results
#        collected_general_results = expand_path(command_dir / "all.results.parquet", expand_command=True),
#        collected_tree_results = expand_path(command_dir / "all.results.trees.parquet", expand_command=True),



include: "rules/per_version_rules.smk"
include: "rules/per_command_rules.smk"

