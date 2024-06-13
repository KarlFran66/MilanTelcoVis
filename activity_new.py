from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_leaflet as dl
from dash_extensions.javascript import arrow_function
from config import data_options, milano_boundary_style, milano_neighbourhoods, popup_style, start_time, end_time, \
    loading_style
from utils import choropleth_style_handle, marks

MAP_STYLE = {"width": "100vw", "height": "95vh", "position": "absolute"}
BUTTON_GROUP_STYLE = {"position": "absolute", "top": "10px", "left": "50px", "zIndex": "1000"}


def activity_new_layout():
    colored_grid = dl.GeoJSON(
        id="grid",
        style=choropleth_style_handle,
        # zoomToBounds=True,
        # zoomToBoundsOnClick=True,
        hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.5))
    ),

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

    choropleth_map = dl.Map(
        id="map",
        center=[45.4642, 9.1900],
        zoom=12.2,
        style=MAP_STYLE,
        children=[
            dl.TileLayer(
                opacity=0.7,
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                # attribution='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            ),

            # colored grid layer
            loading_grid_wrapper,

            # Milano silhouette layer
            dl.GeoJSON(
                id="milano-boundary", data=milano_neighbourhoods.__geo_interface__,
                style=milano_boundary_style
            ),

            # color bar
            html.Div(id='colorbar-container'),

            # box selector
            box_selector
        ])

    button_group = dbc.ButtonGroup(
        [
            dbc.Button("SMS In", id="btn-sms-in", color="secondary", outline=False),
            dbc.Button("SMS Out", id="btn-sms-out", color="success", outline=False),
            dbc.Button("Call In", id="btn-call-in", color="warning", outline=False),
            dbc.Button("Call Out", id="btn-call-out", color="danger", outline=False),
            dbc.Button("Internet", id="btn-internet", color="info", outline=False),
        ],
        style=BUTTON_GROUP_STYLE,
    )

    info_board = html.Div(
        id="info", className="info",
        style={
            "position": "absolute",
            "top": "80px",
            "left": "50px",
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

    range_slider = dbc.Card(
        [
            html.H4("Time Filter", style={"textAlign": "center"}),
            dcc.RangeSlider(
                id='date-range-slider',
                min=start_time,
                max=end_time,
                step=10 * 60 * 1000,
                value=[start_time, end_time],
                marks=marks
            ),
        ],
        style={
            "width": "80%",
            "margin": "0 auto",
            "padding": "15px 35px",
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

    line_plot = dcc.Graph(id="time-series-chart")

    loading_line = dcc.Loading(
        id="loading-map",
        type="graph",
        children=line_plot
    )

    details_window = dbc.Offcanvas(
        children=[
            dcc.Checklist(
                id="checklist",
                options=data_options,
                inline=True,
                value=['sms_in', 'sms_out', 'call_in', 'call_out', 'internet']
            ),
            loading_line
        ],
        id='offcanvas',
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
        dcc.Store(id='feature-store'),
        details_window
    ])
