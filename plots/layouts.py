from dash import html, dcc
import dash_bootstrap_components as dbc


def get_empty_dropdown(label, dropdown_id):
    return html.Div(
        [
            dbc.Label(label),
            dcc.Dropdown(
                id=dropdown_id,
                options=[],
                value=None,
            ),
        ]
    )


layout = html.Div(
    children=[
        dbc.Container(
            [
                html.Div(id="_dummy", hidden=True),
                html.Br(),
                html.H1("RAxML-NG version comparison"),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            get_empty_dropdown(
                                label="RAxML-NG Command", dropdown_id="commandSelector"
                            )
                        ),
                    ]
                ),
                html.Hr(),
                html.Br(),
                html.Div(
                    [
                        html.H3("Summary view"),
                        html.P("The following plots summarize the results for all datasets in one plot "),
                        html.Br(),
                        html.Div(
                            [
                                html.H5("Comparison entire run"),
                                dbc.Col(
                                    get_empty_dropdown(
                                        label="Metric",
                                        dropdown_id="resultMetricSelectorSummary",
                                    ),
                                ),
                                dcc.Graph(id="metricSummaryPlot"),
                            ]
                        ),
                        html.Br(),
                    ]
                ),
                html.Div(
                    [
                        html.H3("Intra-version comparison"),
                        html.P("The following plots summarize the results per tested RAxML-NG version separately. "
                               "That means, that e.g. the RF-Distances between trees always refer to the trees produced"
                               " by the same tool, and the statistical tests were only ran on the trees per tool."),
                        html.Br(),
                        dbc.Col(
                            get_empty_dropdown(
                                label="Dataset", dropdown_id="datasetSelector"
                            ),
                        ),
                        html.Div(
                            [
                                html.H5("Comparison entire run"),
                                dbc.Row(
                                 [
                                  dbc.Col(
                                   [
                                     get_empty_dropdown(
                                        label="Metric",
                                        dropdown_id="resultMetricSelector1",
                                     ),
                                     dcc.Graph(id="metricEntireRunComparison1"),
                                   ]
                                  ),
                                 dbc.Col(
                                   [
                                     get_empty_dropdown(
                                        label="Metric",
                                        dropdown_id="resultMetricSelector2",
                                     ),
                                     dcc.Graph(id="metricEntireRunComparison2"),
                                   ]
                                  ),
                                 ]
                                ),
                            ]
                        ),
                        html.Br(),
                        html.Div(
                            [
                                html.H5("Comparison all ML trees"),
                                dbc.Col(
                                    get_empty_dropdown(
                                        label="Metric",
                                        dropdown_id="resultMetricAllTreesSelector",
                                    ),
                                ),
                                dcc.Graph(id="metricAllTreesComparison"),
                            ]
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H3("Metric-by-Metric view"),
                        html.P("The following plots show by metric comparison."),
                        html.Br(),
                        html.Div(
                            [
                                html.H5("Comparison entire run"),
                                dbc.Row(
                                 [
                                  dbc.Col(
                                      get_empty_dropdown(
                                        label="Metric X",
                                        dropdown_id="resultMetricSelectorX",
                                    ),
                                  ),
                                  dbc.Col(
                                      get_empty_dropdown(
                                        label="Metric Y",
                                        dropdown_id="resultMetricSelectorY",
                                    ),
                                  ),
                                 ],
                                ), 
                                dcc.Graph(id="versionMetricPlot"),
                            ]
                        ),
                        html.Br(),
                    ]
                ),

                # html.Div(
                #     [
                #         html.H3("Inter-version comparison"),
                #         html.P("The following plots summarize the results when comparing all trees across all RAxML-NG tools. "
                #                "That means, that e.g. the RF-Distances between trees always refer the RF-Distances ."),
                #         html.Br(),
                #     ]
                # )
            ]
        )
    ]
)
