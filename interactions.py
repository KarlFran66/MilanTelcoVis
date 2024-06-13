from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function
from config import data_options, milano_boundary_style, milano_neighbourhoods, popup_style, start_time, end_time, \
    loading_style, colorscale_interaction, merged_origin_df, start_date_interactions, end_date_interactions
from utils import choropleth_style_handle, marks_interactions, news_on_each_feature

# MAP_STYLE = {"width": "60vw", "height": "95vh", "position": "absolute"}
BUTTON_GROUP_STYLE = {"position": "absolute", "top": "10px", "left": "70px", "zIndex": "1000"}


def interactions_layout():
    # # prepare classes and colorscale
    # max_value = merged_origin_df['DIS'].max()
    # # print('filtered max_value: ', max_value)
    # step = max_value / (len(colorscale_interaction) - 1)
    # classes = [i * step for i in range(len(colorscale_interaction))]
    # style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)
    #
    # # hideout dict
    # hideout = dict(colorscale=colorscale_interaction, classes=classes, style=style, colorProp='DIS')
    #
    # # categories and children for colorbar
    # ctg = ["{}+".format(int(cls)) for cls in classes[:-1]] + ["{}+".format(int(classes[-1]))]
    # colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale_interaction, width=900, height=30,
    #                                     position="topright")

    colored_grid = dl.GeoJSON(
        id="interactions-grid",
        # style=choropleth_style_handle,
        # zoomToBounds=True,
        # zoomToBoundsOnClick=True,
        hoverStyle=arrow_function(dict(weight=3, color='#666', fillOpacity=0.2)),
        # hideout=hideout,
        # data=merged_origin_df.__geo_interface__
    ),

    loading_grid = dcc.Loading(
        id="interactions-loading-map",
        type="graph",
        children=colored_grid
    )

    loading_grid_wrapper = html.Div(style=loading_style, children=[loading_grid])

    button_group = dbc.ButtonGroup(
        [
            dbc.Button("Telecom Interactions Intensity", id="btn-choropleth", color="success", outline=False, style={"fontSize": "30px"}),
            dbc.Button("Community Detection", id="btn-clustering", color="warning", outline=False, style={"fontSize": "30px"}),
        ],
        style=BUTTON_GROUP_STYLE,
    )

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
            id="interactions-edit-control"
        )
    ], id='interactions-feature-group')

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

    news_markers = dl.GeoJSON(
        id="news-markers-interactions",
        cluster=True,
        superClusterOptions={"radius": 100, "maxZoom": 18},
        zoomToBoundsOnClick=True,
        options={"spiderfyOnMaxZoom": True},
        onEachFeature=news_on_each_feature
    ),

    choropleth_map = dl.Map(
        id="interactions-map",
        center=[45.4642, 9.1900],
        zoom=12.2,
        # style=MAP_STYLE,
        style={"width": "100vw", "height": "95vh", "position": "absolute"},
        children=[
            # layers control
            dl.LayersControl([
                dl.Overlay(satellite_layer, name="Satellite", checked=True),
                dl.Overlay(default_layer, name="Open Street Map", checked=False),
                # Choropleth map
                dl.Overlay(loading_grid_wrapper, name="Telecom Interactions Intensity", checked=True),
                # news markers
                dl.Overlay(news_markers, name="News Markers", checked=False)
            ], position="topleft"),

            # Milano silhouette layer
            dl.GeoJSON(
                id="interactions-milano-boundary", data=milano_neighbourhoods.__geo_interface__,
                style=milano_boundary_style
            ),

            # color bar
            # html.Div(id='interactions-colorbar-container', children=colorbar),
            html.Div(id='interactions-colorbar-container'),

            # box selector
            # box_selector
        ])

    info_board = html.Div(
        id="interactions-info", className="info",
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

    # map_layout = html.Div([
    #     button_group,
    #     info_board,
    #     choropleth_map
    # ], id="map-layout", style={"position": "relative"})

    map_layout = html.Div([
        button_group,
        info_board,
        choropleth_map
    ], id="map-layout", style={"position": "relative", "width": "100vw", "height": "95vh"})

    range_slider = dbc.Card(
        [
            # html.H4("Time Filter", style={"textAlign": "center"}),
            dbc.Row([
                dbc.Col(dcc.RadioItems(
                    id="radioitems-cluster-resolution",
                    options=[
                        {'label': 'Resolution = 1', 'value': '1'},
                        {'label': 'Resolution = 2', 'value': '2'},
                    ],
                    value='1',
                    labelStyle={'display': 'inline-block', 'fontSize': '30px'},
                    inputStyle={"marginRight": "5px", "marginLeft": "20px"}
                ), width=5),
                dbc.Col(html.H4("Time Filter", style={"textAlign": "left", 'fontSize': '35px'}), width=6),
            ], align="center"),
            dcc.RangeSlider(
                id='date-range-slider-interactions',
                min=start_date_interactions,
                max=end_date_interactions,
                step=7 * 24 * 60 * 60 * 1000,
                value=[start_date_interactions, end_date_interactions],
                tooltip={"placement": "top", "always_visible": True, "transform": "formatDate"},
                marks=marks_interactions,
                className='dcc-slider'
            ),
        ],
        style={
            "width": "80%",
            "margin": "0 auto",
            "padding": "15px 40px",
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

    network_plot = html.Iframe(
        # srcDoc=open('path_to_graph1.html', 'r').read(),
        id="network-plot",
        # style={"height": "400px", "width": "100%"},
        style={"height": "450px", "width": "100%", "border": "none"}
    )

    chord_diagram = html.Iframe(
        # srcDoc=open('path_to_graph1.html', 'r').read(),
        id="chord-diagram",
        # style={"height": "400px", "width": "100%"},
        style={"height": "450px", "width": "100%", "border": "none"}
    )

    loading_network = dcc.Loading(
        id="loading-network",
        type="graph",
        children=network_plot,
        style={"height": "100%", "width": "100%"}
    )

    loading_chord = dcc.Loading(
        id="loading-chord",
        type="graph",
        children=chord_diagram,
        style={"height": "100%", "width": "100%"}
    )

    network_card = dbc.Card(
        dbc.CardBody(loading_network, style={"height": "100%", "padding": "0"}),
        className="custom-card"
        # style={"height": "45%", "width": "100%", "padding": "0"}
    )

    chord_card = dbc.Card(
        dbc.CardBody([
            dcc.Checklist(
                id="checklist-chord",
                options=[
                    {'label': 'Include Self-Interactions', 'value': 'Include Self-Interactions'},
                ],
                inline=True,
                value=['Include Self-Interactions'],
                labelStyle={'display': 'inline-block', 'fontSize': '30px'},
                inputStyle={"marginRight": "5px", "marginLeft": "20px"}
            ),
            loading_chord], style={"height": "100%", "padding": "0"}),
        className="custom-card"
        # style={"height": "45%", "width": "100%", "padding": "0"}
    )

    details_window = dbc.Offcanvas(
        children=[
            network_card,
            chord_card
        ],
        id='offcanvas-interactions',
        # title='Interactions between Clusters',
        title=html.Span('Interactions between Clusters', className='custom-offcanvas-title', style={'fontSize': '40px'}),
        is_open=False,
        placement='end',
        backdrop=False,
        scrollable=True,
        style={"width": "40vw", "height": "100vh", "display": "flex", "flexDirection": "column"}
    )

    return html.Div([
        map_layout,
        dcc.Store(id='interactions-type-store'),
        range_slider,
        # dcc.Store(id='feature-store'),
        details_window
    ])
