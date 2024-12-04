import pathlib
import yaml

from plotly import graph_objects as go


config = yaml.load(open("config.yaml"), yaml.Loader)

RESULTS_BASE = pathlib.Path(config["outdir"])
DATASETS = sorted([d for d in RESULTS_BASE.iterdir() if d.is_dir()])
CMD_MAPPINGS = None

# plot definitions
TEMPLATE = "plotly_white"

HVLABEL = { "bgcolor": "white", "font": {"color": "black"} }   

VERSION_COMPARISON_PLOT_METRICS_SUMMARY = {
    "speedup": go.Box,
    "runtime": go.Scatter,
    "avgBootstrapSupport" : go.Scatter,
    "absoluteLogLikelihoodDiff": go.Box,
    "relativeLogLikelihoodDiff": go.Box,
    "relativeRFDistanceMLTrees": go.Scatter,
    "numberOfInferredTrees": go.Scatter,
    "uniqueTopologiesMLTrees": go.Scatter,
    "isPlausible" : go.Box,
    "pythiaDifficultyScore" : go.Scatter,
    "bestAICScore": go.Scatter,
    "bestAICcScore": go.Scatter,
    "bestBICScore": go.Scatter,
}

VERSION_COMPARISON_PLOT_METRICS_ENTIRE_RUN = {
    "bestLogLikelihood": go.Scatter,
    "runtime": go.Scatter,
    "numberOfInferredTrees": go.Scatter,
    "uniqueTopologiesMLTrees": go.Scatter,
    "relativeRFDistanceMLTrees": go.Scatter,
    "absoluteRFDistanceMLTrees": go.Scatter,
    "avgBootstrapLogLikelihood" : go.Scatter,
    "avgBootstrapSupport" : go.Scatter,
    "pythiaDifficultyScore" : go.Scatter,
    "bestAICScore": go.Scatter,
    "bestAICcScore": go.Scatter,
    "bestBICScore": go.Scatter,
}

VERSION_COMPARISON_PLOT_METRICS_ALL_TREES = {
    "logLikelihood": go.Box,
    "plausible": go.Bar,
    "pKH": go.Box,
    "pKH_significant": go.Bar,
    "pSH": go.Box,
    "pSH_significant": go.Bar,
    "pWKH": go.Box,
    "pWKH_significant": go.Bar,
    "pWSH": go.Box,
    "pWSH_significant": go.Bar,
    "pAU": go.Box,
    "pAU_significant": go.Bar,
}
