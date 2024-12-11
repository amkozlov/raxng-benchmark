import re
import subprocess

from tempfile import TemporaryDirectory

from .custom_types import *
from .utils import *


def get_raxmlng_best_llh(raxmlng_file: FilePath) -> float:
    STR = "Final LogLikelihood:"
    return get_single_value_from_file(raxmlng_file, STR)


def get_raxmlng_likelihoods(raxmlng_file: FilePath) -> List[float]:
    # [00:00:27] [worker #4] ML tree search #13, logLikelihood: -6485.304526
    llh_regex = re.compile("\[\d+:\d+:\d+\]\s*(\[worker\s+#\d+\])?\s+ML\s+tree\s+search\s+#(\d+),\s+logLikelihood")
    likelihoods = []

    for line in read_file_contents(raxmlng_file):
        if "logLikelihood" in line:
            m = llh_regex.search(line)
            if m:
                tree_id = int(m.groups()[1])
                _, llh = line.rsplit(":", 1)
                llh = float(llh)
                likelihoods.append((tree_id, llh))

    likelihoods.sort()

    return [llh for _, llh in likelihoods]

def get_raxmlng_bootstrap_likelihoods(raxmlng_file: FilePath) -> List[float]:
    # [00:00:00] [worker #0] Bootstrap tree #1, logLikelihood: -2746.271209
    llh_regex = re.compile("\[\d+:\d+:\d+\]\s*(\[worker\s+#\d+\])?\s+Bootstrap\s+tree\s+#(\d+),\s+logLikelihood")
    likelihoods = []

    for line in read_file_contents(raxmlng_file):
        if "logLikelihood" in line:
            m = llh_regex.search(line)
            if m:
                tree_id = int(m.groups()[1])
                _, llh = line.rsplit(":", 1)
                llh = float(llh)
                likelihoods.append((tree_id, llh))

    likelihoods.sort()

    return [llh for _, llh in likelihoods]

def get_raxmlng_bootstrap_supports(support_file: FilePath) -> List[float]:
    supports = re.findall("\)(\d+):", open(support_file).readline().strip())

    return [float(x) for x in supports]

def get_raxmlng_time_from_line(line: str) -> float:
    # two cases now:
    # either the run was cancelled an rescheduled
    if "restarts" in line:
        # line looks like this: "Elapsed time: 5562.869 seconds (this run) / 91413.668 seconds (total with restarts)"
        _, right = line.split("/")
        value = right.split(" ")[1]
        return float(value)

    # ...or the run ran in one sitting...
    else:
        # line looks like this: "Elapsed time: 63514.086 seconds"
        value = line.split(" ")[2]
        return float(value)


def get_raxmlng_elapsed_time(log_file: FilePath) -> float:
    content = read_file_contents(log_file)

    for line in content:
        if "Elapsed time:" not in line:
            continue
        else:
            return get_raxmlng_time_from_line(line)

    raise ValueError(
        f"The given input file {log_file} does not contain the elapsed time."
    )


def get_raxmlng_ic_scores(raxmlng_file: FilePath) -> Tuple[float, float, float]:
    # AIC score: 1640560.457503 / AICc score: 1640565.191951 / BIC score: 1642721.583409
    ic_regex = re.compile("AIC score:\s+([0-9.]+)\s+/\s+AICc score:\s+([0-9.]+)\s+/\s+BIC score:\s+([0-9.]+)")
    likelihoods = []

    aic_score = None
    aicc_score = None
    bic_score = None

    for line in read_file_contents(raxmlng_file):
        if line.startswith("AIC score:"):
            m = ic_regex.search(line)
            if m:
                aic_score = float(m.groups()[0])
                aicc_score = float(m.groups()[1])
                bic_score = float(m.groups()[2])
                break

    return aic_score, aicc_score, bic_score

def get_raxmlng_pythia_score(raxmlng_file: FilePath) -> float:
    #[00:00:00] Predicted difficulty: 0.07
    pythia_regex = re.compile("\[\d+:\d+:\d+\]\s*Predicted difficulty:\s([0-9.]+)")

    for line in read_file_contents(raxmlng_file):
        if "difficulty:" in line:
            m = pythia_regex.search(line)
            if m:
                return float(m.groups()[0])
   
    return None

# returns #taxa, #sites, #patterns, #partitions
def get_raxmlng_msa_dimensions(raxmlng_file: FilePath) -> Tuple[int, int, int, int]:
    #[00:00:00] Loaded alignment with 994 taxa and 5533 sites
    site_regex = re.compile("\[\d+:\d+:\d+\]\s*Loaded alignment with\s([0-9]+) taxa and ([0-9]+) sites")
    #Alignment comprises 1 partitions and 3363 patterns
    part_regex = re.compile("Alignment comprises\s([0-9]+) partitions and ([0-9]+) patterns")

    num_taxa = num_sites = num_patterns = num_partitions = None

    for line in read_file_contents(raxmlng_file):
        if "Loaded alignment" in line:
            m = site_regex.search(line)
            if m:
                num_taxa = int(m.groups()[0])
                num_sites = int(m.groups()[1])
        elif "Alignment comprises" in line:
            m = part_regex.search(line)
            if m:
                num_partitions = int(m.groups()[0])
                num_patterns = int(m.groups()[1])
                break 

    return num_taxa, num_sites, num_patterns, num_partitions


def raxmlng_rfdist(raxmlng: Executable, trees_file: FilePath) -> Tuple[float, float, float]:
    with TemporaryDirectory() as tmpdir:
        prefix = pathlib.Path(tmpdir) / "rfdist"

        cmd = [
            raxmlng,
            "--rfdist",
            trees_file,
            "--prefix",
            prefix,
        ]

        subprocess.check_output(cmd)

        log_file = pathlib.Path(f"{prefix}.raxml.log")

        abs_rfdist = None
        rel_rfdist = None
        num_topos = None

        for line in read_file_contents(log_file):
            if "Average absolute RF distance in this tree set:" in line:
                abs_rfdist = get_value_from_line(
                    line, "Average absolute RF distance in this tree set:"
                )
            elif "Average relative RF distance in this tree set:" in line:
                rel_rfdist = get_value_from_line(
                    line, "Average relative RF distance in this tree set:"
                )
            elif "Number of unique topologies in this tree set:" in line:
                num_topos = get_value_from_line(
                    line, "Number of unique topologies in this tree set:"
                )

        if abs_rfdist is None or rel_rfdist is None or num_topos is None:
            raise ValueError(f"Error parsing raxml-ng logfile {log_file.name}.")

        return num_topos, rel_rfdist, abs_rfdist
