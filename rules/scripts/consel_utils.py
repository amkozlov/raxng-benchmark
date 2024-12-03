from snakemake.shell import shell

from .custom_types import *
import os


def consel_statistical_tests(
    consel_dir: str,
    raxml_ng: Executable,
    cmd_extra: str,
    msa: FilePath,
    model: str,
    sitelh_output_prefix : FilePath,
    sitelh_output: FilePath,
    consel_output: FilePath,
    mlTrees: FilePath,
    sitelh_snakelog: FilePath,
    consel_snakelog: FilePath
):

    # consel executables
    makermt = f"{consel_dir}/makermt"
    consel = f"{consel_dir}/consel"
    catpv = f"{consel_dir}/catpv"

    # compute per-site log-likelihoods
    shell(
        f"{raxml_ng} --sitelh {cmd_extra} "
        f"--msa {msa} --model {model} "
        f"--tree {mlTrees} --prefix {sitelh_output_prefix} "
        f"--seed 0 --redo --lh-epsilon 1.0 "
        f"> {sitelh_snakelog} 2>&1"
    )

    # make a copy of RAxML-NG output
    # consel takes as input a file with suffix '.sitelh'
    # and not '.siteLH'
    # RAxML-NG output: all.raxml.siteLH
    # consel input: allraxml.sitelh
    folder_path = os.path.dirname(sitelh_output)
    outifle = sitelh_output.split('/')[-1]

    first_part, _ , remaining_part = outifle.partition('.')
    sitelh_output_consel_cmptbl = first_part + remaining_part.lower()
    sitelh_output_consel_cmptbl_prfx = sitelh_output_consel_cmptbl.rpartition('.')[0]
    
    sitelh_output_consel_cmptbl = f"{folder_path}/{sitelh_output_consel_cmptbl}"
    sitelh_output_consel_cmptbl_prfx = f"{folder_path}/{sitelh_output_consel_cmptbl_prfx}"
    
    # copy
    shell(
        f"cp {sitelh_output} {sitelh_output_consel_cmptbl}"
    )

    # consel
    shell(
        f"{makermt} --puzzle {sitelh_output_consel_cmptbl} "
        f"> {consel_snakelog} 2>&1"
    )

    shell(
        f"{consel} {sitelh_output_consel_cmptbl_prfx} "
        f"> {consel_snakelog} 2>&1"
    )

    shell(
        f"{catpv} {sitelh_output_consel_cmptbl_prfx} "
        f"> {consel_output}"
    )