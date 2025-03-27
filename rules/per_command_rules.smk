import numpy as np
import pandas as pd
import numpy as np

from scripts.raxmlng_utils import *

from scripts.iqtree_utils import iqtree_statistical_tests
from scripts.iqtree_statstest_parser import get_iqtree_results

from scripts.consel_utils import consel_statistical_tests
from scripts.consel_statstest_parser import get_consel_results

versions_only = [str(key) for key in raxmlng_versions]
#versions_only = [f.name for f in command_dir.iterdir() if f.is_dir()]

rule collect_trees_per_command:
    input:
        # I know it's naive, but it's the only command that worked properly
        raxml_logiles = [str(command_dir) + "/" + vers + "/" + vers + ".raxml.log" for vers in versions_only],
        raxml_mltrees = [str(command_dir) + "/" + vers + "/" + vers + ".raxml.mlTrees" for vers in versions_only]
        
    output:
        mlTrees = command_dir / "all.mlTrees",
        versions = command_dir / "all.versions",
        bestTree = command_dir / "all.bestTree"
        
    run:
        best_tree = None
        best_llh = -np.inf
        
        versions_list = []
        mlTrees_list = []

        for logfile,mlTrees_file in zip(input.raxml_logiles, input.raxml_mltrees):

            mlTrees = open(mlTrees_file, 'r').readlines()
            mlTrees = [tree.rstrip().replace(" ", "") for tree in mlTrees]
            if mlTrees[-1] == "": mlTrees = mlTrees[:-1]
            mlTrees_list += mlTrees
            
            raxml_version = logfile.split("/")[-1].split(".raxml.")[0]
            versions_list += [raxml_version for _ in range(len(mlTrees))]
            
            likelihoods = get_loglhs_from_logfile(logfile, len(mlTrees))
            idx = np.argmax(likelihoods)
            if likelihoods[idx] > best_llh:
                best_llh = likelihoods[idx]
                best_tree = mlTrees[idx]
        
        with open(output.mlTrees, "w") as mlt:
            for elem in mlTrees_list:
                mlt.write(f"{elem}\n")

        open(output.bestTree, "w").write(best_tree.strip())
        
        with open(output.versions, "w") as vf:
            for elem in versions_list:
                vf.write(elem + "\n")


rule consel_significance_tests_per_command:
    input:
        mlTrees = rules.collect_trees_per_command.output.mlTrees
    output:
        raxml_log = command_dir / "all.raxml.log",
        sitelh_out = command_dir / "all.raxml.siteLH",
        consel_out = command_dir / "all.consel"
    log:
        sitelh_snakelog = command_dir / "all.sitelh.snakelog",
        consel_snakelog = command_dir / "all.consel.snakelog"
    params:
        prefix = str(command_dir / "all")
    run:
        msa = msas[wildcards.msa]
        raxml_ng, *extra = config["raxmlng_evaluation_bin"]

        consel_statistical_tests(
            consel_dir=config['consel_dir'][0],
            raxml_ng=raxml_ng,
            cmd_extra=extra[0] if len(extra) > 0 else "",
            msa=msa,
            model=models[msa],
            sitelh_output_prefix=params.prefix,
            sitelh_output=output.sitelh_out,
            consel_output=output.consel_out,
            mlTrees=input.mlTrees,
            sitelh_snakelog = log.sitelh_snakelog,
            consel_snakelog = log.consel_snakelog
        )


rule collect_results_per_command:
    input:
        general_results_per_version = expand_path(rules.collect_results_per_raxmlng_version.output.general_results, expand_command=False, expand_dataset=False),
        consel_file = rules.consel_significance_tests_per_command.output.consel_out,
        raxml_log = rules.consel_significance_tests_per_command.output.raxml_log,
        mltrees_file = rules.collect_trees_per_command.output.mlTrees,
        versions_file = rules.collect_trees_per_command.output.versions

    output:
        collected_results = command_dir / "collected.results.parquet",
        general_results = command_dir / "all.results.parquet"
        
    run:
        consel_results_reduced = get_consel_results(input.consel_file, 
                                                    input.raxml_log, 
                                                    input.mltrees_file, 
                                                    input.versions_file, 
                                                    output.collected_results)
        
        all_results = pd.concat([pd.read_parquet(f) for f in input.general_results_per_version], ignore_index=True).reset_index(drop=True)

        # pull reference version from config file
        baseline_ver = config['reference_version'][0]

        baseline_version_line = all_results[all_results["version"] == baseline_ver].reset_index(drop=True)
        baseline_runtime = baseline_version_line["runtime"][0]
        baseline_loglh = baseline_version_line["bestLogLikelihood"][0]
        baseline_bic = baseline_version_line["bestBICScore"][0]

        all_results = all_results.assign(speedup = baseline_runtime / all_results["runtime"], 
                                         absoluteLogLikelihoodDiff = all_results["bestLogLikelihood"] - baseline_loglh,
                                         absoluteBICScoreDiff =  all_results["bestBICScore"] - baseline_bic)

        all_results = all_results.assign(relativeLogLikelihoodDiff = -1 * all_results["absoluteLogLikelihoodDiff"] / baseline_loglh,
                                         relativeBICScoreDiff = -1 * all_results["absoluteBICScoreDiff"] / baseline_bic)

        print(all_results)

        all_results.to_parquet(output.general_results)

"""
rule iqtree_significance_tests_per_command:
    input:
        bestTree = rules.collect_trees_per_command.output.bestTree,
        mlTrees = rules.collect_trees_per_command.output.mlTrees
    output:
        results = command_dir / "all.iqtree"
    log:
        snakelog = command_dir / "all.iqtree.snakelog"
    params:
        prefix = str(command_dir / "all")
    run:
        msa = msas[wildcards.msa]
        iqtree, *extra = config["iqtree"]

        iqtree_statistical_tests(
            iqtree=iqtree,
            msa=msa,
            model=models[msa],
            output_prefix=params.prefix,
            mlTrees=input.mlTrees,
            bestTree=input.bestTree,
            snakelog=log.snakelog
        )

"""
