import pandas as pd
import plotly.express as px

from dash.dependencies import Output, Input
from plotly import graph_objects as go

from app import app
from definitions import *

CWAY=["red", "green", "blue"]
TFMT="%r"

def get_plot_options(plot_type, df, metric, dataset = None):
    plot_data = df[metric]

    if plot_type == go.Box:
        plot_options = {
            "y": plot_data,
            "boxpoints": "all",
            "showlegend": False,
        }
        xtitle = "RAxML-NG version"
        ytitle = metric
    elif plot_type == go.Bar:
        plot_options = {
            "x": plot_data.value_counts().index,
            "y": plot_data.value_counts().values,
            "showlegend": True
        }
        xtitle = metric
        ytitle = "Number of trees"
    elif plot_type == go.Scatter and dataset:
        plot_options = {
            "y": plot_data,
            "marker": dict(size=10),
            "showlegend": True,
        }
        xtitle = "meaningless x-Axis"
        ytitle = metric
    elif plot_type == go.Scatter:
        plot_options = {
            "y": plot_data,
            "boxpoints": "all",
            "showlegend": False,
            "hoverlabel": { "bgcolor": "white" }
        }
        xtitle = "RAxML-NG version"
        ytitle = metric
    else:
        plot_options = {}
        xtitle = ""
        ytitle = ""

    return plot_options, xtitle, ytitle


@app.callback(
    Output("metricEntireRunComparison1", "figure"),
    Input("datasetSelector", "value"),
    Input("commandSelector", "value"),
    Input("resultMetricSelector1", "value")
)
def plot_per_command_comparison(dataset, command, metric):
    results_dir = RESULTS_BASE / dataset / command
    raxmlng_versions = [d for d in results_dir.iterdir() if d.is_dir()]

    fig = go.Figure()

    for version in raxmlng_versions:
        df = pd.read_parquet(version / (version.name + ".results.parquet"))
        plot_type = VERSION_COMPARISON_PLOT_METRICS_ENTIRE_RUN[metric]

        plot_options, xtitle, ytitle = get_plot_options(plot_type, df, metric, dataset)

        fig.add_trace(
            plot_type(
                name=version.name,
                **plot_options
            )
        )

        fig.update_xaxes(title=xtitle)
        fig.update_yaxes(title=ytitle, tickformat=TFMT)

    fig.update_layout(template=TEMPLATE, colorway=CWAY, hoverlabel=HVLABEL)
    return fig

@app.callback(
    Output("metricEntireRunComparison2", "figure"),
    Input("datasetSelector", "value"),
    Input("commandSelector", "value"),
    Input("resultMetricSelector2", "value")
)
def plot_per_command_comparison(dataset, command, metric):
    results_dir = RESULTS_BASE / dataset / command
    raxmlng_versions = [d for d in results_dir.iterdir() if d.is_dir()]

    fig = go.Figure()

    for version in raxmlng_versions:
        df = pd.read_parquet(version / (version.name + ".results.parquet"))
        plot_type = VERSION_COMPARISON_PLOT_METRICS_ENTIRE_RUN[metric]

        plot_options, xtitle, ytitle = get_plot_options(plot_type, df, metric, dataset)

        fig.add_trace(
            plot_type(
                name=version.name,
                **plot_options
            )
        )

        fig.update_xaxes(title=xtitle)
        fig.update_yaxes(title=ytitle, tickformat=TFMT)
        
    fig.update_layout(template=TEMPLATE, colorway=CWAY, hoverlabel=HVLABEL)

    return fig


@app.callback(
    Output("metricAllTreesComparison", "figure"),
    Input("datasetSelector", "value"),
    Input("commandSelector", "value"),
    Input("resultMetricAllTreesSelector", "value")
)

def plot_per_command_comparison(dataset, command, metric):
    results_dir = RESULTS_BASE / dataset / command
    raxmlng_versions = [d for d in results_dir.iterdir() if d.is_dir()]

    fig = go.Figure()

    for version in raxmlng_versions:
        df = pd.read_parquet(version / (version.name + ".consel.results.trees.parquet"))
        plot_type = VERSION_COMPARISON_PLOT_METRICS_ALL_TREES[metric]

        plot_options, xtitle, ytitle = get_plot_options(plot_type, df, metric, dataset)

        fig.add_trace(
            plot_type(
                name=version.name,
                **plot_options
            )
        )

        fig.update_xaxes(title=xtitle)
        fig.update_yaxes(title=ytitle, tickformat=TFMT)

    fig.update_layout(template=TEMPLATE, colorway=CWAY, hoverlabel=HVLABEL)
    return fig


@app.callback(
    Output("metricSummaryPlot", "figure"),
    Input("commandSelector", "value"),
    Input("resultMetricSelectorSummary", "value")
)
def plot_per_command_summary(command, metric):
    datasets = [d for d in RESULTS_BASE.iterdir() if d.is_dir() and (d / command / "all.results.parquet").is_file()]

    fig = go.Figure()

    df = pd.concat([pd.read_parquet(dset / command / "all.results.parquet").assign(dataset=dset.name) for dset in datasets], ignore_index=True).reset_index(drop=True)

    plot_type = VERSION_COMPARISON_PLOT_METRICS_SUMMARY[metric]

    fig = px.box(df, x = "version", y = metric, color="version", points="all", hover_data=["dataset"] )
    fig.update_yaxes(tickformat=TFMT)

    fig.update_layout(template=TEMPLATE, colorway=CWAY, hoverlabel=HVLABEL)
    return fig

@app.callback(
    Output("versionMetricPlot", "figure"),
    Input("commandSelector", "value"),
    Input("resultMetricSelectorX", "value"),
    Input("resultMetricSelectorY", "value")
)
def plot_per_metric(command, xmetric, ymetric):
    datasets = [d for d in RESULTS_BASE.iterdir() if d.is_dir() and (d / command / "all.results.parquet").is_file()]

    fig = go.Figure()

    df = pd.concat([pd.read_parquet(dset / command / "all.results.parquet").assign(dataset=dset.name) for dset in datasets], ignore_index=True).reset_index(drop=True)

    plot_type = VERSION_COMPARISON_PLOT_METRICS_SUMMARY[ymetric]

#    xmetric = "pythiaDifficultyScore"
    fig = px.box(df, x = xmetric, y = ymetric, color="version", points="all", hover_data=["dataset"] )
    fig.update_yaxes(tickformat=TFMT)

    fig.update_layout(template=TEMPLATE, colorway=CWAY, hoverlabel=HVLABEL)
    return fig

