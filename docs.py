from dash import html
import dash_bootstrap_components as dbc

def docs_layout():
    return html.Div(
        [
            dbc.Row(
                dbc.Col(
                    html.H1("MilanTelcoVis Documents", style={"textAlign": "center", "marginTop": "20px"}),
                    width=12
                )
            )
        ]
    )
