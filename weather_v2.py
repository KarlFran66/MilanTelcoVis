from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from config import data_options, milano_boundary_style, milano_neighbourhoods, popup_style, start_time, end_time, \
    loading_style, start_time_date, end_time_date, activity_weather
from utils import choropleth_style_handle, marks, marks_v3, news_on_each_feature, bivariate_style_handle, generate_bivariate_color_matrix


MAP_STYLE = {"width": "100%", "height": "100%"}
BUTTON_GROUP_STYLE = {"position": "absolute", "top": "10px", "left": "50px", "zIndex": "1000"}


def weather_layout_v2():
    bivariate_grid_v2 = dl.GeoJSON(
        id="bivariate-grid-v2",
        style=bivariate_style_handle,
        hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray='')),
    )

    loading_bivariate_grid_v2 = dcc.Loading(
        id="loading-bivariate-grid-v2",
        type="graph",
        children=bivariate_grid_v2
    )

    loading_bivariate_grid_wrapper_v2 = html.Div(style=loading_style, children=[loading_bivariate_grid_v2])

    colored_weather_grid_v2 = dl.GeoJSON(
        id="grid-weather-v2",
        style=choropleth_style_handle,
        # zoomToBounds=True,
        # zoomToBoundsOnClick=True,
        hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.5))
    ),

    loading_weather_grid_v2 = dcc.Loading(
        id="loading-map-weather-v2",
        type="graph",
        children=colored_weather_grid_v2
    )

    loading_weather_grid_wrapper_v2 = html.Div(style=loading_style, children=[loading_weather_grid_v2])

    colored_activity_grid_v2 = dl.GeoJSON(
        id="grid-activity-v2",
        style=choropleth_style_handle,
        # zoomToBounds=True,
        # zoomToBoundsOnClick=True,
        hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.5))
    ),

    loading_activity_grid_v2 = dcc.Loading(
        id="loading-map-activity-v2",
        type="graph",
        children=colored_activity_grid_v2
    )

    loading_activity_grid_wrapper_v2 = html.Div(style=loading_style, children=[loading_activity_grid_v2])

    satellite_layer = dl.TileLayer(
        opacity=0.8,
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attribution=''
        # attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    ),

    default_layer = dl.TileLayer(
        opacity=0.5,
        attribution=''
    ),

    bivariate_colorscale_container = html.Div(id='bivariate-colorscale-container', style={
        'position': 'absolute',
        'top': '10px',
        'right': '10px',
        'background-color': 'rgba(255, 255, 255, 0.9)',
        'padding': '10px',
        "borderRadius": "8px",
        "border": "3px solid black",
        'z-index': '1000'
    })

    sensor_markers = dl.LayerGroup(id='sensor-markers')

    bivariate_map_v2 = dl.Map(
        id="map-weather-v2",
        center=[45.4742, 9.1900],
        zoom=12.2,
        style=MAP_STYLE,
        children=[
            # layers control
            dl.LayersControl([
                # base map layers
                dl.Overlay(satellite_layer, name="Satellite", checked=True),
                dl.Overlay(default_layer, name="Open Street Map", checked=False),

                # bivariate map layer
                dl.Overlay(loading_bivariate_grid_wrapper_v2, name="Bivariate Intensity", checked=True),

                # weather layer
                dl.Overlay(loading_weather_grid_wrapper_v2, name="Weather Feature Intensity", checked=False),

                # activity layer
                dl.Overlay(loading_activity_grid_wrapper_v2, name="Telecom Activity Intensity", checked=False),
            ], position="topleft"),

            # Milano silhouette layer
            dl.GeoJSON(
                id="milano-boundary-weather-v2", data=milano_neighbourhoods.__geo_interface__,
                style=milano_boundary_style
            ),

            bivariate_colorscale_container,

            sensor_markers
        ])

    info_board_weather_v2 = html.Div(
        id="info-weather-v2", className="info",
        style={
            "position": "absolute",
            "top": "10px",
            "left": "70px",
            "borderRadius": "8px",
            "border": "3px solid black",
            "backgroundColor": "rgba(255, 255, 255, 0.9)",
            "zIndex": "1000"
        }
    )

    map_layout_weather_v2 = html.Div([
        info_board_weather_v2,
        bivariate_map_v2
    ], style={"position": "relative", "height": "100%", "width": "100%"})

    parallel_coordinates_v2 = dcc.Graph(id="parallel-coordinates-v2", style={'height': '400px', 'width': '100%'})

    loading_parallel_coordinates_v2 = dcc.Loading(
        id="loading-parallel-coordinates-v2",
        type="graph",
        children=parallel_coordinates_v2
    )

    scatter_v2 = dcc.Graph(id="scatter-v2", style={'height': '428px', 'width': '100%'})

    loading_scatter_v2 = dcc.Loading(
        id="loading-scatter-v2",
        type="graph",
        children=scatter_v2
    )

    line_v2 = dcc.Graph(id="line-v2", style={'height': '390px', 'width': '100%'})

    loading_line_v2 = dcc.Loading(
        id="loading-line-v2",
        type="graph",
        children=line_v2
    )

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Row(dcc.RadioItems(
                    id="radioitems-weather-v2-activity",
                    options=[
                        {'label': 'SMS In', 'value': 'sms_in'},
                        {'label': 'SMS Out', 'value': 'sms_out'},
                        {'label': 'Call In', 'value': 'call_in'},
                        {'label': 'Call Out', 'value': 'call_out'},
                        {'label': 'Internet', 'value': 'internet'},
                    ],
                    value='sms_in',
                    labelStyle={'display': 'inline-block', 'fontSize': '27px'},
                    inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                )),
                dbc.Row(dcc.RadioItems(
                    id="radioitems-weather-v2-weather",
                    options=[
                        {'label': 'Temperature', 'value': 'Temperature'},
                        {'label': 'Relative Humidity', 'value': 'Relative Humidity'},
                        {'label': 'Wind Speed', 'value': 'Wind Speed'},
                        {'label': 'Precipitation', 'value': 'Precipitation'},
                    ],
                    value='Precipitation',
                    labelStyle={'display': 'inline-block', 'fontSize': '27px'},
                    inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                )),
            ], width={'size': 4}),
        ], style={'marginBottom': '10px'}),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        dcc.RadioItems(
                            id="radioitems-select-days-v2",
                            options=[
                                {'label': 'All Days', 'value': 'All Days'},
                                {'label': 'Only Weekdays', 'value': 'Only Weekdays'},
                                {'label': 'Only Weekends', 'value': 'Only Weekends'}
                            ],
                            value='Only Weekdays',
                            labelStyle={'display': 'inline-block', 'fontSize': '27px'},
                            inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                        ),
                        dcc.Checklist(
                            id="checklist-log-v2",
                            options=[
                                {'label': 'Log Scale Activity', 'value': 'Log Scale Activity'},
                                {'label': 'Log Scale Weather', 'value': 'Log Scale Weather'},
                                {'label': 'Filter Zeros', 'value': 'Filter Zeros'}
                            ],
                            inline=True,
                            value=['Log Scale Weather', 'Filter Zeros'],
                            labelStyle={'display': 'inline-block', 'fontSize': '27px'},
                            inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                        ),
                        loading_parallel_coordinates_v2
                    ]),
                ], style={'height': '43vh', 'marginBottom': '3px'}), width={'size': 4},

            ),
            dbc.Col(dbc.Card(
                dbc.CardBody(
                    map_layout_weather_v2,
                    style={"height": "100%", "padding": "0"}
                ),
                style={"height": "43vh", "border": "1px solid #ccc", "borderRadius": "8px", 'marginBottom': '0px'}
            ), width={'size': 8}),
        ], style={'marginBottom': '3px'}),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    loading_scatter_v2
                ])
            ]), width={'size': 4},
                style={'height': '27vh', 'marginBottom': '3px'}
            ),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    dcc.RadioItems(
                        id="radioitems-select-sensor-v2",
                        labelStyle={'display': 'inline-block', 'fontSize': '27px'},
                        inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                    ),
                    loading_line_v2
                ])
            ]), width={'size': 8},
                style={'height': '23vh', 'marginBottom': '3px'}
            ),
        ]),
    ], style={"height": "90vh"})

