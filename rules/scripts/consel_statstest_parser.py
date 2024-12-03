import regex
import warnings
import numpy as np
import re
import pandas as pd
import os

TEST_NAMES = ["p-KH", "p-SH", "p-WKH", "p-WSH", "p-AU"]

def get_relevant_section(input_file):
    
    content = open(input_file,'r').readlines()
    
    return content[3:-1]

def get_loglhs_from_logfile(raxml_log, n):
    
    loglhs = np.zeros(n, dtype=float)

    mylines = open(raxml_log, 'r').readlines()

    for line in mylines:
        line = line.rstrip()
        if ", final logLikelihood:" in line:
            idx = int(line.split("#")[-1].split(",")[0])
            _loglh = float(line.split(" ")[-1])
            loglhs[idx-1] = _loglh


    assert np.all(loglhs != 0), "Error while collecting log-likelihoods from all.raxml.log"

    return loglhs

def process_line(line):

    line = line.rstrip()
    line = re.split(r'\s+', line.strip())
    idx = int(line[2])
    _obs = float(line[3])
    
    _pAU = 0.5 if _obs == 0 else float(line[4])
    _pKH = 0.5 if _obs == 0 else float(line[9])
    _pWKH = 0.5 if _obs == 0 else float(line[11])
    _pSH = 0.5 if _obs == 0 else float(line[10])
    _pWSH = 0.5 if _obs == 0 else float(line[12])

    return idx ,_pKH, _pWKH, _pSH, _pWSH, _pAU


def get_pvalues(section, n):
    
    assert len(section) == n, "The length of the relevant section in consel file is not equal to the num of trees?"

    pKH = np.zeros(n, dtype=float)
    pWKH = np.zeros(n, dtype=float)
    pSH = np.zeros(n, dtype=float)
    pWSH = np.zeros(n, dtype=float)
    pAU = np.zeros(n, dtype=float)

    for line in section:
        line = line.rstrip()
        idx ,_pKH, _pWKH, _pSH, _pWSH, _pAU = process_line(line)

        pKH[idx - 1] = _pKH
        pWKH[idx - 1] = _pWKH
        pSH[idx - 1] = _pSH
        pWSH[idx - 1] = _pWSH
        pAU[idx - 1] = _pAU
    
    return pKH, pWKH, pSH, pWSH, pAU


def get_significance(pKH, pWKH, pSH, pWSH, pAU, n):

    pKH_significant = [False]*n
    pWKH_significant = [False]*n
    pSH_significant = [False]*n
    pWSH_significant = [False]*n
    pAU_significant = [False]*n
    plausible = [False]*n

    for idx in range(n):
        pKH_significant[idx] = pKH[idx] >= 0.05
        pWKH_significant[idx] = pWKH[idx] >= 0.05
        pSH_significant[idx] = pSH[idx] >= 0.05
        pWSH_significant[idx] = pWSH[idx] >= 0.05
        pAU_significant[idx] = pAU[idx] >= 0.05
        
        plausible[idx] = pKH_significant[idx] and pWKH_significant[idx] and \
                         pSH_significant[idx] and pWSH_significant[idx] and \
                         pAU_significant[idx]

    return pKH_significant, pWKH_significant, pSH_significant, pWSH_significant, pAU_significant, plausible


def get_consel_results(consel_file, raxml_log, mlTrees_file, versions_file, outfile):
    
    cmd_dir = os.path.dirname(outfile)

    versions = open(versions_file, 'r').readlines()
    versions = [v.rstrip() for v in versions]
    
    mlTrees = open(mlTrees_file, 'r').readlines()
    mlTrees = [tree.rstrip().replace(" ", "") for tree in mlTrees]
    if mlTrees[-1] == "": mlTrees = mlTrees[:-1]
    num_trees = len(mlTrees)

    loglhs = get_loglhs_from_logfile(raxml_log, num_trees)

    is_best = [False]*num_trees
    is_best[np.argmax(loglhs)] = True

    section = get_relevant_section(consel_file)
    
    pKH, pWKH, pSH, pWSH, pAU = get_pvalues(section, num_trees)

    pKH_significant, pWKH_significant, pSH_significant, \
    pWSH_significant, pAU_significant, plausible = get_significance(pKH, pWKH, pSH, pWSH, pAU, num_trees)

    out_df = pd.DataFrame({
        "newick": mlTrees,
        "logLikelihood": loglhs,
        "isBest": is_best,
        "plausible": plausible,
        "pKH": pKH,
        "pKH_significant": pKH_significant,
        "pWKH": pWKH,
        "pWKH_significant": pWKH_significant,
        "pSH": pSH,
        "pSH_significant": pSH_significant,
        "pWSH": pWSH,
        "pWSH_significant": pWSH_significant,
        "pAU": pAU,
        "pAU_significant": pAU_significant,
        "version": versions
    })

    out_df.to_parquet(outfile)

    versions_set = list(set(versions))

    best_tree_dict = []

    for _version in versions_set:
        version_outfile = f"{cmd_dir}/{_version}/{_version}.consel.results.trees.parquet"
        out_df_version = out_df[out_df.version == _version]
        out_df_version.index = list(range(len(out_df_version)))
        
        idx = np.argmax(out_df_version.logLikelihood)
        out_df_version.loc[idx, "isBest"] = True
        out_df_version.to_parquet(version_outfile)
        
        out_df_version.pop("newick")
        version_dict = out_df_version.to_dict(orient='records')
        best_tree_dict.append(version_dict[idx])

        # correct summary "{version}.results.parquet" file
        tmp_file = f"{cmd_dir}/{_version}/{_version}.results.parquet"
        tmp_data = pd.read_parquet(tmp_file)
        tmp_data.loc[0, "bestLogLikelihood"] = version_dict[idx]["logLikelihood"]
        tmp_data["version"] = _version
        tmp_data["isPlausible"] = 1 if version_dict[idx]["plausible"] else 0
        tmp_data.to_parquet(tmp_file)

    return best_tree_dict