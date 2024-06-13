from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function
from config import data_options, milano_boundary_style, milano_neighbourhoods, popup_style, start_time, end_time, \
    loading_style, start_time_date, end_time_date
from utils import choropleth_style_handle, marks, marks_v3, news_on_each_feature, bivariate_style_handle

MAP_STYLE = {"width": "100vw", "height": "95vh", "position": "absolute"}
BUTTON_GROUP_STYLE = {"position": "absolute", "top": "10px", "left": "50px", "zIndex": "1000"}


def weather_layout():
    colored_weather_grid = dl.GeoJSON(
        id="weather-grid-weather",
        style=choropleth_style_handle,
        hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.5))
    ),

    loading_weather_grid = dcc.Loading(
        id="loading-weather-map-weather",
        type="graph",
        children=colored_weather_grid
    )

    loading_weather_grid_wrapper = html.Div(style=loading_style, children=[loading_weather_grid])

    bivariate_grid = dl.GeoJSON(
        id="bivariate-grid",
        style=bivariate_style_handle,
        hoverStyle=arrow_function(dict(weight=5, color='#666', dashArray='')),
    )

    loading_bivariate_grid = dcc.Loading(
        id="loading-bivariate-grid",
        type="graph",
        children=bivariate_grid
    )

    loading_bivariate_grid_wrapper = html.Div(style=loading_style, children=[loading_bivariate_grid])

    colored_grid = dl.GeoJSON(
        id="grid-weather",
        style=choropleth_style_handle,
        # zoomToBounds=True,
        # zoomToBoundsOnClick=True,
        hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.5))
    ),

    loading_grid = dcc.Loading(
        id="loading-map-weather",
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
            id="edit-control-weather"
        )
    ], id='feature-group-weather')

    satellite_layer = dl.TileLayer(
        opacity=0.8,
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        # attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    ),

    default_layer = dl.TileLayer(
        opacity=0.5,
        # url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        # attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    ),

    news_markers = dl.GeoJSON(
        id="news-markers-weather",
        cluster=True,
        superClusterOptions={"radius": 100, "maxZoom": 18},
        zoomToBoundsOnClick=True,
        options={"spiderfyOnMaxZoom": True},
        onEachFeature=news_on_each_feature
    ),

    choropleth_map = dl.Map(
        id="map-weather",
        center=[45.4642, 9.1900],
        zoom=12.2,
        style=MAP_STYLE,
        children=[
            # layers control
            dl.LayersControl([
                dl.Overlay(satellite_layer, name="Satellite", checked=True),
                dl.Overlay(default_layer, name="Open Street Map", checked=False),
                # weather map
                dl.Overlay(loading_weather_grid_wrapper, name="Weather Intensity", checked=False),
                dl.Overlay(loading_bivariate_grid_wrapper, name="Bivariate Intensity", checked=True),
                # Choropleth map
                dl.Overlay(loading_grid_wrapper, name="Telecom Activity Intensity", checked=False),
                # news markers
                dl.Overlay(news_markers, name="News Markers", checked=False)
            ], position="topleft"),

            # Milano silhouette layer
            dl.GeoJSON(
                id="milano-boundary-weather", data=milano_neighbourhoods.__geo_interface__,
                style=milano_boundary_style
            ),

            # color bar
            html.Div(id='colorbar-container-weather'),
            html.Div(id='weather-colorbar-container-weather'),

            # box selector
            box_selector
        ])

    button_group = dbc.ButtonGroup(
        [
            dbc.Button("SMS In", id="btn-sms-in-weather", color="secondary", outline=False),
            dbc.Button("SMS Out", id="btn-sms-out-weather", color="success", outline=False),
            dbc.Button("Call In", id="btn-call-in-weather", color="warning", outline=False),
            dbc.Button("Call Out", id="btn-call-out-weather", color="danger", outline=False),
            dbc.Button("Internet", id="btn-internet-weather", color="info", outline=False),
        ],
        style=BUTTON_GROUP_STYLE,
    )

    info_board = html.Div(
        id="info-weather", className="info",
        style={
            "position": "absolute",
            "top": "80px",
            "left": "70px",
            "borderRadius": "8px",
            "border": "3px solid black",
            "backgroundColor": "rgba(255, 255, 255, 0.7)",
            "zIndex": "1000"
        }
    )

    map_layout = html.Div([
        button_group,
        info_board,
        choropleth_map
    ], style={"position": "relative"})

    marks_time = {i: f"{i // 6:02d}:{(i % 6) * 10:02d}" for i in range(0, 144, 6)}

    range_slider = dbc.Card(
        [
            dbc.Row([
                dbc.Col(dcc.RadioItems(
                    id="radioitems-filter-weather",
                    options=[
                        {'label': 'per hour', 'value': 'hour'},
                        {'label': 'per 10 minutes', 'value': 'minute'},
                    ],
                    value='hour',
                    labelStyle={'display': 'inline-block'},
                    inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                ), width=5),
                dbc.Col(html.H4("Time Filter", style={"textAlign": "left"}), width=7),
            ], align="center"),
            dcc.RangeSlider(
                id='date-range-slider-weather',
                min=start_time_date,
                max=end_time_date,
                step=24 * 60 * 60 * 1000,
                value=[start_time_date, end_time_date],
                tooltip={"placement": "top", "always_visible": True, "transform": "formatDate"},
                marks=marks_v3,
                className='dcc-slider'
            ),
            dcc.Slider(
                id='start-date-time-slider-weather',
                min=0,
                max=143,
                step=1,
                value=0,
                tooltip={"placement": "top", "always_visible": True, "transform": "formatTime"},
                marks=marks_time,
                included=False,
                className='dcc-slider'
            ),
            dcc.Slider(
                id='end-date-time-slider-weather',
                min=0,
                max=143,
                step=1,
                value=0,
                tooltip={"placement": "top", "always_visible": True, "transform": "formatTime"},
                marks=marks_time,
                included=False,
                className='dcc-slider'
            )
        ],
        style={
            "width": "80%",
            "margin": "0 auto",
            "padding": "5px 40px",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
            "borderRadius": "8px",
            "border": "3px solid black",
            "position": "absolute",
            "bottom": "10px",
            "left": "50%",
            "transform": "translateX(-50%)",
            "backgroundColor": "rgba(255, 255, 255, 0.7)",
            "zIndex": "2000"
        },
        className="mb-3",
    )

    line_plot = dcc.Graph(id="time-series-chart-weather")

    loading_line = dcc.Loading(
        id="loading-map-weather",
        type="graph",
        children=line_plot
    )

    details_window = dbc.Offcanvas(
        children=[
            dcc.Checklist(
                id="checklist-weather",
                options=data_options,
                inline=True,
                value=['sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
            ),

            dcc.RadioItems(
                id="radioitems-weather",
                options=[
                    {'label': 'per hour', 'value': 'hour'},
                    {'label': 'per 10 minutes', 'value': 'minute'},
                ],
                value='hour'
            ),

            loading_line
        ],
        id='offcanvas-weather',
        title='curve test',
        is_open=False,
        placement='end',
        backdrop=False,
        scrollable=True,
        style={"width": "40vw"}
    )

    return html.Div([
        map_layout,
        range_slider,
        dcc.Store(id='feature-store-weather'),
        details_window,
        # dcc.Interval(
        #     id='interval-v4',
        #     disabled=False,
        #     interval=1*10000,  # update every 10 seconds
        #     n_intervals=0,
        #     max_intervals=1890
        # )
    ])
