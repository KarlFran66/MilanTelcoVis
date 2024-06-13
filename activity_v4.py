from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function
from config import data_options, milano_boundary_style, milano_neighbourhoods, popup_style, start_time, end_time, \
    loading_style, start_time_date, end_time_date
from utils import choropleth_style_handle, marks, marks_v3, news_on_each_feature

MAP_STYLE = {"width": "100vw", "height": "95vh", "position": "absolute"}
BUTTON_GROUP_STYLE = {"position": "absolute", "top": "10px", "left": "70px", "zIndex": "1000"}


def activity_v4_layout():
    colored_grid = dl.GeoJSON(
        id="grid-v4",
        style=choropleth_style_handle,
        # zoomToBounds=True,
        # zoomToBoundsOnClick=True,
        hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.5))
    ),

    loading_grid = dcc.Loading(
        id="loading-map-v4",
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
            edit={"edit": False, "remove": True},
            id="edit-control-v4"
        )
    ], id='feature-group-v4')

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
        id="news-markers-v4",
        cluster=True,
        superClusterOptions={"radius": 100, "maxZoom": 18},
        zoomToBoundsOnClick=True,
        options={"spiderfyOnMaxZoom": True},
        onEachFeature=news_on_each_feature
    ),

    choropleth_map = dl.Map(
        id="map-v4",
        center=[45.4642, 9.1900],
        zoom=12.2,
        style=MAP_STYLE,
        children=[
            # layers control
            dl.LayersControl([
                dl.Overlay(satellite_layer, name="Satellite", checked=True),
                dl.Overlay(default_layer, name="Open Street Map", checked=False),
                # Choropleth map
                dl.Overlay(loading_grid_wrapper, name="Telecom Activity Intensity", checked=True),
                # news markers
                dl.Overlay(news_markers, name="News Markers", checked=False)
            ], position="topleft"),

            # Milano silhouette layer
            dl.GeoJSON(
                id="milano-boundary-v4", data=milano_neighbourhoods.__geo_interface__,
                style=milano_boundary_style
            ),

            # color bar
            html.Div(id='colorbar-container-v4'),

            # box selector
            box_selector
        ])

    button_group = dbc.ButtonGroup(
        [
            dbc.Button("SMS In", id="btn-sms-in-v4", color="secondary", outline=False, style={"fontSize": "30px"}),
            dbc.Button("SMS Out", id="btn-sms-out-v4", color="warning", outline=False, style={"fontSize": "30px"}),
            dbc.Button("Call In", id="btn-call-in-v4", color="success", outline=False, style={"fontSize": "30px"}),
            dbc.Button("Call Out", id="btn-call-out-v4", color="danger", outline=False, style={"fontSize": "30px"}),
            dbc.Button("Internet", id="btn-internet-v4", color="info", outline=False, style={"fontSize": "30px"}),
        ],
        style=BUTTON_GROUP_STYLE,
    )

    info_board = html.Div(
        id="info-v4", className="info",
        style={
            "position": "absolute",
            "top": "80px",
            "left": "70px",
            "borderRadius": "8px",
            "border": "3px solid black",
            "backgroundColor": "rgba(255, 255, 255, 0.9)",
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
                    id="radioitems-filter-v4",
                    options=[
                        {'label': 'per hour', 'value': 'hour'},
                        {'label': 'per 10 minutes', 'value': 'minute'},
                    ],
                    value='hour',
                    labelStyle={'display': 'inline-block', 'fontSize': '30px'},
                    inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                ), width=5),
                dbc.Col(html.H4("Time Filter", style={"textAlign": "left", "fontSize": "35px"}), width=7),
            ], align="center"),
            dcc.RangeSlider(
                id='date-range-slider-v4',
                min=start_time_date,
                max=end_time_date,
                step=24 * 60 * 60 * 1000,
                value=[start_time_date, end_time_date],
                tooltip={"placement": "top", "always_visible": True, "transform": "formatDate"},
                marks=marks_v3,
                className='dcc-slider'
            ),
            dcc.Slider(
                id='start-date-time-slider-v4',
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
                id='end-date-time-slider-v4',
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
            "backgroundColor": "rgba(255, 255, 255, 0.9)",
            "zIndex": "2000"
        },
        className="mb-3",
    )

    line_plot = dcc.Graph(id="time-series-chart-v4")

    loading_line = dcc.Loading(
        id="loading-map-v4",
        type="graph",
        children=line_plot
    )

    line_card = dbc.Card(
        dbc.CardBody([
            dcc.Checklist(
                id="checklist-v4",
                options=data_options,
                inline=True,
                value=['sms_in', 'sms_out', 'call_in', 'call_out', 'internet'],
                labelStyle={'display': 'inline-block', 'fontSize': '30px'},
                inputStyle={"marginRight": "5px", "marginLeft": "20px"}
            ),

            dcc.RadioItems(
                id="radioitems-v4",
                options=[
                    {'label': 'per hour', 'value': 'hour'},
                    {'label': 'per 10 minutes', 'value': 'minute'},
                ],
                value='hour',
                labelStyle={'display': 'inline-block', 'fontSize': '30px'},
                inputStyle={"marginRight": "5px", "marginLeft": "20px"}
            ),

            loading_line,],

            style={"height": "100%", "padding": "0"}),
        className="custom-card"
        # style={"height": "45%", "width": "100%", "padding": "0"}
    )

    details_window = dbc.Offcanvas(
        children=[
            # dcc.Checklist(
            #     id="checklist-v4",
            #     options=data_options,
            #     inline=True,
            #     value=['sms_in', 'sms_out', 'call_in', 'call_out', 'internet'],
            #     labelStyle={'display': 'inline-block'},
            #     inputStyle={"marginRight": "5px", "marginLeft": "20px"}
            # ),
            #
            # dcc.RadioItems(
            #     id="radioitems-v4",
            #     options=[
            #         {'label': 'per hour', 'value': 'hour'},
            #         {'label': 'per 10 minutes', 'value': 'minute'},
            #     ],
            #     value='hour',
            #     labelStyle={'display': 'inline-block'},
            #     inputStyle={"marginRight": "5px", "marginLeft": "20px"}
            # ),
            #
            # loading_line
            line_card
        ],
        id='offcanvas-v4',
        # title='curve test',
        title=html.Span('Telecom Activity Details', className='custom-offcanvas-title', style={'fontSize': '40px'}),
        is_open=False,
        placement='end',
        backdrop=False,
        scrollable=True,
        style={"width": "40vw"}
    )

    return html.Div([
        map_layout,
        range_slider,
        dcc.Store(id='feature-store-v4'),
        details_window,
        # dcc.Interval(
        #     id='interval-v4',
        #     disabled=False,
        #     interval=1*10000,  # update every 10 seconds
        #     n_intervals=0,
        #     max_intervals=1890
        # )
    ])
