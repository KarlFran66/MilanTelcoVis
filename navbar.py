import dash_bootstrap_components as dbc
from dash import html

def Navbar():
    color_mode_switch = html.Span(
        [
            dbc.Label(className="fa fa-moon", html_for="switch"),
            dbc.Switch(id="switch", value=True, className="d-inline-block ms-1", persistence=True),
            dbc.Label(className="fa fa-sun", html_for="switch"),
        ]
    )

    navbar = dbc.NavbarSimple(
        children=[
            # dbc.DropdownMenu(
            #     children=[
            #         dbc.DropdownMenuItem("Telecom Activity (V1)", href="/activity_v1"),
            #         dbc.DropdownMenuItem("Telecom Activity (V2)", href="/activity_v2"),
            #         dbc.DropdownMenuItem("Telecom Activity (V3)", href="/activity_v3"),
            #         dbc.DropdownMenuItem("Telecom Activity (V4)", href="/activity_v4"),
            #     ],
            #     nav=True,
            #     in_navbar=True,
            #     label="Telecom Activity",
            # ),
            dbc.NavItem(dbc.NavLink("Telecom Activity", href="/activity_v4")),
            dbc.NavItem(dbc.NavLink("Telecom Interactions", href="/interactions")),
            dbc.NavItem(dbc.NavLink("Weather", href="/weather_v2")),
            dbc.NavItem(dbc.NavLink("Docs", href="/docs")),
            # dbc.DropdownMenu(
            #     children=[
            #         dbc.DropdownMenuItem("Weather (V1)", href="/weather_v1"),
            #         dbc.DropdownMenuItem("Weather (V2)", href="/weather_v2"),
            #     ],
            #     nav=True,
            #     in_navbar=True,
            #     label="Weather",
            # ),
            dbc.NavItem(color_mode_switch)
        ],
        brand="MilanTelcoVis",
        style={'fontSize': '1.7em'},
        brand_href="#",
        color="dark",
        dark=True
    )
    return navbar
