import json
from dash.dependencies import Output, Input

from app import app
from definitions import *


@app.callback(
    Output("datasetSelector", "options"),
    Output("datasetSelector", "value"),
    Input("commandSelector", "value"),
)
def populate_dataset_selector(cmd):
    options = []
    for dataset in DATASETS:
      cmd_dir = dataset / cmd
      if pathlib.Path(cmd_dir).exists():
        options.append({"label": dataset.name, "value": dataset.name})
    return options, options[0]["value"]


@app.callback(
    Output("commandSelector", "options"),
    Output("commandSelector", "value"),
    Input("_dummy", "children"),
)
def populate_command_selector(_):
    with open(RESULTS_BASE / "cmd_mapping.json") as f:
        CMD_MAPPING = json.load(f)
#    dataset_dir = RESULTS_BASE / dataset

#    options = [{"label": CMD_MAPPING[cmd.name], "value": cmd.name} for cmd in dataset_dir.iterdir()]
    options = [{"label": CMD_MAPPING[cmd], "value": cmd} for cmd in CMD_MAPPING]
    return options, options[0]["value"]


@app.callback(
    Output("resultMetricSelector1", "options"),
    Output("resultMetricSelector1", "value"),
    Input("_dummy", "children"),
)
def populate_entire_run_comparison_metric_selector(_):
    options = [{"label": col, "value": col} for col in VERSION_COMPARISON_PLOT_METRICS_ENTIRE_RUN]
    return options, options[0]["value"]


@app.callback(
    Output("resultMetricSelector2", "options"),
    Output("resultMetricSelector2", "value"),
    Input("_dummy", "children"),
)
def populate_entire_run_comparison_metric_selector(_):
    options = [{"label": col, "value": col} for col in VERSION_COMPARISON_PLOT_METRICS_ENTIRE_RUN]
    return options, options[0]["value"]

@app.callback(
    Output("resultMetricAllTreesSelector", "options"),
    Output("resultMetricAllTreesSelector", "value"),
    Input("_dummy", "children"),
)
def populate_all_tree_comparison_metric_selector(_):
    options = [{"label": col, "value": col} for col in VERSION_COMPARISON_PLOT_METRICS_ALL_TREES]
    return options, options[0]["value"]

@app.callback(
    Output("resultMetricSelectorSummary", "options"),
    Output("resultMetricSelectorSummary", "value"),
    Input("_dummy", "children"),
)
def populate_summary_metric_selector(_):
    options = [{"label": col, "value": col} for col in VERSION_COMPARISON_PLOT_METRICS_SUMMARY]
    return options, options[0]["value"]

@app.callback(
    Output("resultMetricSelectorX", "options"),
    Output("resultMetricSelectorX", "value"),
    Input("_dummy", "children"),
)
def populate_summary_metric_selector_x(_):
    options = [{"label": col, "value": col} for col in VERSION_COMPARISON_PLOT_METRICS_SUMMARY]
    return options, options[0]["value"]

@app.callback(
    Output("resultMetricSelectorY", "options"),
    Output("resultMetricSelectorY", "value"),
    Input("_dummy", "children"),
)
def populate_summary_metric_selector_y(_):
    options = [{"label": col, "value": col} for col in VERSION_COMPARISON_PLOT_METRICS_SUMMARY]
    return options, options[0]["value"]

