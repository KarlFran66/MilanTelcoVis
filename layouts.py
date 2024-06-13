from dash import dcc, html
from navbar import Navbar
import dash_bootstrap_components as dbc


def global_layout():
    layout = html.Div([
        dcc.Location(id='url', refresh=False),
        # dbc.Container(
        #     [
                dbc.Row(dbc.Col(Navbar(), width=12)),
                dbc.Row(dbc.Col(html.Div(id='page-content'), width=12))
            # ],
            # fluid=True,
            # style={'height': '100vh'}
        # )
    ])
    return layout
