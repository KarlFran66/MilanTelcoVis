from dash import html
from dash_extensions.javascript import assign
import pandas as pd
from dash.exceptions import PreventUpdate
import matplotlib.colors as mcolors
from config import start_time, end_time, start_time_date, end_time_date, start_date_interactions, end_date_interactions


# popup style for news markers
# news_on_each_feature = assign("""function(feature, layer) {
#     if (feature.properties && feature.properties.popup) {
#         layer.bindPopup(feature.properties.popup);
#     }
# }""")
news_on_each_feature = assign("""function(feature, layer) {
    if (feature.properties && feature.properties.popup) {
        layer.bindPopup(feature.properties.popup, {className: 'custom-tooltip'});
    }
}""")

# function for getting info for info board
def get_info(selected_data_label, selected_data, feature=None):
    header = [html.H4("Milano Traffic Data (" + selected_data_label + ")", style={"textAlign": "left", 'fontSize': '30px'})]
    if not feature:
        return header + [html.P("Hoover over a grid", style={"textAlign": "left", 'fontSize': '25px'})]
    return header + [
    html.Div(
        html.B("Grid id: {}".format(feature["properties"]['square_id']), style={"fontSize": "25px"}),
        style={"marginBottom": "10px"}
    ),
    html.Div(
        html.Span("Proportional Value: {:.2f}".format(feature["properties"][selected_data]), style={"fontSize": "25px"}),
        style={"marginBottom": "10px"}
    )
    ]

def get_info_weather(activity_feature, weather_feature, feature=None):
    header = [html.H4("Telecom Activity and Weather (" + activity_feature + ", " + weather_feature + ")", style={"textAlign": "left", 'fontSize': '30px'})]
    if not feature:
        return header + [html.P("Hoover over a grid", style={"textAlign": "left", 'fontSize': '25px'})]
    return header + [
                        html.Div(
                            html.B("Grid id: {}".format(feature["properties"]['square_id']), style={"fontSize": "25px"}),
                            style={"marginBottom": "10px"}
                        ),
                        html.Div(
                            html.Span("Proportional telecom activity: {:.2f}".format(feature["properties"][activity_feature]), style={"fontSize": "25px"}),
                            style={"marginBottom": "10px"}
                        ),
                        html.Div(
                            html.Span("Average temperature: {:.2f}℃".format(feature["properties"]['Temperature']), style={"fontSize": "25px"}),
                            style={"marginBottom": "10px"}
                        ),
                        html.Div(
                            html.Span("Average relative humidity: {:.2f}%".format(feature['properties']['Relative Humidity']), style={"fontSize": "25px"}),
                            style={"marginBottom": "10px"}
                        ),
                        html.Div(
                            html.Span("Average wind speed: {:.2f}m/s".format(feature['properties']['Wind Speed']), style={"fontSize": "25px"}),
                            style={"marginBottom": "10px"}
                        ),
                        html.Div(
                            html.Span("Total precipitation: {:.2f}mm".format(feature['properties']['Precipitation']), style={"fontSize": "25px"}),
                            style={"marginBottom": "15px"}
                        )
                    ]

def generate_bivariate_color_matrix(colorscale, num_colors, row_label, col_label):
    colorscale_length = len(colorscale)

    top_labels = [
        html.Td(''),
        html.Td(f'{col_label} ➡', colSpan=num_colors, style={'text-align': 'center', 'font-weight': 'bold', "fontSize": "20px"})
    ]

    rows = [html.Tr(top_labels)]
    for i in range(num_colors):
        row = [
            html.Td(f'{row_label} ⬇' if i == num_colors // 2 else '', style={'text-align': 'center', 'font-weight': 'bold', "fontSize": "20px"}),
            *[
                html.Td(style={
                    'background-color': colorscale[(i * num_colors + j) % colorscale_length],
                    'width': '20px', 'height': '20px',
                    'border': 'none'})
                for j in range(num_colors)
            ]
        ]
        rows.append(html.Tr(row))

    return html.Table(
        rows,
        style={'border-collapse': 'collapse', 'border': 'none'}
    )

def interactions_get_info(feature=None):
    header = [html.H4("Milano Telecom Interactions Data", style={"textAlign": "left", 'fontSize': '30px'})]
    if not feature:
        return header + [html.P("Hoover over a grid", style={"textAlign": "left", 'fontSize': '25px'})]
    return header + [
        html.Div(
            html.B("Grid ID: {}".format(feature["properties"]['node']), style={"fontSize": "25px"}),
            style={"marginBottom": "10px"}
        ),
        html.Div(
            html.Span("Proportional Aggregated Telecom Intensity: {:.2f}".format(feature["properties"]['DIS']), style={"fontSize": "25px"}),
            style={"marginTop": "10px"}
        )
    ]

def interactions_get_info_clusters(feature=None):
    header = [html.H4("Result of Community Detection", style={"textAlign": "left", 'fontSize': '30px'})]
    if not feature:
        return header + [html.P("Hoover over a cluster", style={"textAlign": "left", 'fontSize': '25px'})]
    return header + [html.B("Cluster No: {}".format(feature["properties"]['cluster_label']), style={"textAlign": "left", 'fontSize': '25px'})]

# JavaScript function to define color for each grid
choropleth_style_handle = assign("""function(feature, context){
    const {classes, colorscale, style, colorProp} = context.hideout;
    const value = feature.properties[colorProp];
    for (let i = classes.length - 1; i >= 0; --i) {
        if (value >= classes[i]) {
            style.fillColor = colorscale[i];
            break;
        }
    }
    return style;
}""")

bivariate_style_handle = assign("""function(feature, context){
    const {activity_feature, weather_feature, blended_bivariate_colorscale, classes, classes_precipitation, precipitation_colorscale, style} = context.hideout;
    const call_out_value = feature.properties[activity_feature];
    const precipitation_value = feature.properties[weather_feature];

    let ci_idx = 0; 
    let pi_idx = 0;  

    for (let i = classes.length - 1; i >= 0; i--) {
        if (call_out_value >= classes[i]) {
            ci_idx = i;
            break;  
        }
    }

    for (let i = classes_precipitation.length - 1; i >= 0; i--) {
        if (precipitation_value >= classes_precipitation[i]) {
            pi_idx = i;
            break;  
        }
    }

    const bivariate_color_idx = ci_idx * precipitation_colorscale.length + pi_idx;

    style.fillColor = blended_bivariate_colorscale[bivariate_color_idx];  

    return style;
}""")

# cluster_style_handle = assign("""function(feature, context){
#     style = {fillColor: "#FF0000", color: "white", weight: 1, fillOpacity: 0.7};
#     return style;
# }""")

cluster_style_handle = assign("""function(feature, context){
    const {colors, style, length} = context.hideout;
    const value = feature.properties['cluster_label'];
    for (let i = 0; i < length; ++i) {
        if (value == i) {
            style.fillColor = colors[i];
            break;
        }
    }
    return style;
}""")

# change int to datetime64[ns]
def int_to_date(int_date):
    return pd.to_datetime(int_date, unit='ms')

def int_to_date_v3(int_date):
    return pd.to_datetime(int_date, unit='ms').normalize().strftime('%Y-%m-%d')

marks = {i: int_to_date(i) for i in range(start_time, end_time+1, 7*24*60*60*1000)}  # seven days as slider interval
marks[end_time] = int_to_date(end_time)  # add end time

# marks_v3 = {i: int_to_date_v3(i) for i in range(start_time_date, end_time_date+1, 7*24*60*60*1000)}  # seven days as slider interval
# marks_v3[end_time_date] = int_to_date_v3(end_time_date) + '               ' # add end time

marks_v3 = {
    i: {
        'label': str(int_to_date_v3(i)),
        'style': {'white-space': 'nowrap'}
    } for i in range(start_time_date, end_time_date+1, 7*24*60*60*1000)
}

marks_v3[end_time_date] = {
    'label': str(int_to_date_v3(end_time_date)),
    'style': {'white-space': 'nowrap'}
}

marks_interactions = {
    i: {
        'label': str(int_to_date_v3(i)),
        'style': {'white-space': 'nowrap'}
    } for i in range(start_date_interactions, end_date_interactions+1, 7*24*60*60*1000)
}

marks_interactions[end_time_date] = {
    'label': str(int_to_date_v3(end_date_interactions)),
    'style': {'white-space': 'nowrap'}
}

def hex_to_rgb(value):
    """Convert hex to an RGB tuple."""
    value = value.strip("#")
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def blend_colors(color1, color2):
    """Blend two hex colors together."""
    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)
    # Take the average of each RGB component and scale to [0, 1]
    blended_rgb = tuple(((c1 + c2) / 2) / 255 for c1, c2 in zip(rgb1, rgb2))
    # Convert blended RGB back to hex
    return mcolors.to_hex(blended_rgb)
