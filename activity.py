from dash import dcc, html
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function
import dash_bootstrap_components as dbc
from config import data_options, milano_boundary_style, milano_neighbourhoods, popup_style, start_time, end_time, \
    loading_style
from utils import choropleth_style_handle, marks
import time
from navbar import Navbar


def activity_layout():
    map_start_time = time.time()

    colored_grid = dl.GeoJSON(id="grid",
                              style=choropleth_style_handle,
                              # zoomToBounds=True,
                              # zoomToBoundsOnClick=True,
                              hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.5))),

    loading_grid = dcc.Loading(
        id="loading-map",
        type="graph",
        children=colored_grid
    )

    loading_grid_wrapper = html.Div(style=loading_style, children=[loading_grid])

    box_selector = dl.FeatureGroup([
        dl.EditControl(
            position='topleft',
            draw={
                "rectangle": True,
                "polyline": False,
                "polygon": True,
                "circle": False,
                "marker": False,
                "circlemarker": False
            },
            edit={"edit": True, "remove": True},
            id="edit-control"
        )
    ], id='feature-group')

    choropleth_map = dl.Map(id="map", center=[45.4642, 9.1900], zoom=12.2,
                            style={'width': '100%', 'height': '100%', 'position': 'relative', "zIndex": "1000"},
                            children=[

                                dl.TileLayer(
                                    opacity=0.7,
                                    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                    attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
                                ),

                                # colored grid layer
                                loading_grid_wrapper,

                                # Milano silhouette layer
                                dl.GeoJSON(id="milano-boundary", data=milano_neighbourhoods.__geo_interface__,
                                           style=milano_boundary_style),

                                # color bar
                                html.Div(id='colorbar-container'),

                                # box selector
                                box_selector
                            ])

    dropdown_box = html.Div([
        dcc.Dropdown(
            id="data-selector",
            options=data_options,
            multi=False,
            value="sms_in",  # default value
            style={"width": "200px"}
        )
    ], style={"padding": "10px", "width": "12%", "float": "left"})

    map_layout = html.Div([
        # dropdown box
        dropdown_box,

        # choropleth map
        html.Div([
            choropleth_map,

            # info board
            html.Div(id="info", className="info",
                     style={"position": "absolute", "top": "10px", "right": "10px", "zIndex": "1000"})

        ], style={"width": "88%", "height": "90vh", "float": "right"}),

    ], style={"display": "flex", "flexDirection": "row", "height": "90vh"})

    map_end_time = time.time()

    map_rendering_time = map_end_time - map_start_time

    range_slider = html.Div([
        dcc.RangeSlider(
            id='date-range-slider',
            min=start_time,
            max=end_time,
            step=10 * 60 * 1000,
            value=[start_time, end_time],
            marks=marks
        )
    ], className="range-slider-container")

    line_plot = dcc.Graph(id="time-series-chart")

    loading_line = dcc.Loading(
        id="loading-map",
        type="graph",
        children=line_plot
    )

    # loading_line_wrapper = html.Div(style=loading_style, children=[loading_line])

    details_window = html.Div(id="popup", className="modal", style=popup_style, children=[
        html.Div(className="modal-content", children=[
            html.Span(id="close", className="close", children="×"),
            dcc.Checklist(
                id="checklist",
                options=data_options,
                inline=True,
                value=['sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
            ),
            # dcc.Graph(id="time-series-chart"),
            loading_line
        ]),
    ])

    return html.Div([
        map_layout,

        # details window
        details_window,

        range_slider
    ])

