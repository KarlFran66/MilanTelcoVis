import geopandas as gpd
import pandas as pd
import dask_geopandas as dgpd
import dask.dataframe as dd
from dask.distributed import Client
from datetime import datetime
import vaex
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


cmap_red = plt.get_cmap('Greens', 5)
colors_red = cmap_red(np.arange(5))
colorscale_red = [matplotlib.colors.rgb2hex(color) for color in colors_red]

cmap_blue = plt.get_cmap('Purples', 5)
colors_blue = cmap_blue(np.arange(5))
colorscale_blue = [matplotlib.colors.rgb2hex(color) for color in colors_blue]

# options for dropdown box
data_options = [
    {"label": "SMS-In", "value": "sms_in"},
    {"label": "SMS-Out", "value": "sms_out"},
    {"label": "Call-In", "value": "call_in"},
    {"label": "Call-Out", "value": "call_out"},
    {"label": "Internet", "value": "internet"},
]

# style for the boundary of the city of milano
milano_boundary_style = {
    'color': '#0078A8',  # blue
    'weight': 2,
    'fillColor': '#0078A8',  # blue
    'fillOpacity': 0.0
}

# style for details popup window
popup_style = {
    "display": "block",
    "position": "fixed",
    "top": "50%",
    "right": "-100%",
    "transform": "translateY(-50%)",
    "width": "50%",
    "minWidth": "300px",
    "height": "auto",
    "maxHeight": "80%",
    "overflow": "auto",
    "padding": "20px",
    "background": "#fff",
    "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
    "borderRadius": "10px",
    "zIndex": 1000,
    "transition": "right 0.5s",
    "fontSize": "16px",
    "color": "#333",
}

# colorscale for activity colorbar
# colorscale = ['#FFEDA0', '#FED976', '#FDB84C', '#FC7E2A', '#FA6008', '#F84E02', '#E53B00', '#C02B00', '#9D1B00',
#               '#7A0B00']

colorscale = [
    '#FFEDA0', '#FFEE90', '#FED976', '#FEC462', '#FDB84C', '#FD9F37',
    '#FC7E2A', '#FB6916', '#FA6008', '#F75705', '#F84E02', '#F14501',
    '#E53B00', '#D03300', '#C02B00', '#AB2200', '#9D1B00', '#891200',
    '#7A0B00', '#6B0400'
]

# colorscale for interactions colorbar
colorscale_interaction = [
    '#f2e5f9', '#e7d1ec', '#ddbee0', '#d3abd4', '#c998c8', '#bf85be',
    '#b572b5', '#ab63ad', '#a056a6', '#964ea0', '#8b4999', '#814592',
    '#77408b', '#6d3c84', '#63387c', '#583274', '#4e266b', '#431a62',
    '#380d56', '#2d004b'
]

# style for loading spinner
loading_style = {
    "position": "absolute",
    "top": "50%",
    "left": "50%",
    "transform": "translate(-50%, -50%)",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "zIndex": "2000"
}


def load_milano_neighbourhoods_data():
    # milano_neighbourhoods = gpd.read_file('../Test Data/milano_neighbourhoods.geojson')
    milano_neighbourhoods = gpd.read_file('milano_neighbourhoods.geojson')
    return milano_neighbourhoods


milano_neighbourhoods = load_milano_neighbourhoods_data()

# print('milano_neighbourhoods is:')
# print(len(milano_neighbourhoods))
# print(milano_neighbourhoods.info())
# print(milano_neighbourhoods)


def load_milano_grid_data():
    df_grid = gpd.read_file('milano-grid.geojson')

    grid_x_centers = [point.centroid.x for point in df_grid.geometry]
    grid_y_centers = [point.centroid.y for point in df_grid.geometry]

    x_min, x_max = min(grid_x_centers), max(grid_x_centers)
    y_min, y_max = min(grid_y_centers), max(grid_y_centers)

    n_points = 100

    x_step = (x_max - x_min) / (n_points - 1)
    y_step = (y_max - y_min) / (n_points - 1)

    x_coords = np.arange(x_min, x_max + x_step, x_step)
    y_coords = np.arange(y_min, y_max + y_step, y_step)

    x_coords = np.round(x_coords, 5)
    y_coords = np.round(y_coords, 5)

    x_coords = x_coords[:n_points]
    y_coords = y_coords[:n_points]
    return df_grid, x_coords, y_coords


milano_grid, x_coords, y_coords = load_milano_grid_data()
# print('milano_grid is:')
# print(len(milano_grid))
# print(milano_grid)

def load_aggregated_interaction_data():
    aggregated_interaction = pd.read_parquet('test_aggregated_interaction_one_day.parquet')
    return aggregated_interaction


origin_aggregated_df_interaction = load_aggregated_interaction_data()
merged_origin_df = milano_grid.merge(origin_aggregated_df_interaction, left_on='cellId', right_on='id_origin',
                                     how='inner')
merged_origin_df = gpd.GeoDataFrame(merged_origin_df, geometry='geometry')
merged_origin_df = merged_origin_df.drop(columns=['cellId'])

# def load_interaction_clustered_data():
# clustered_interaction = gpd.read_parquet('test_detected_community.parquet')


# clustered_interaction = gpd.GeoDataFrame(clustered_interaction, geometry='geometry')

# def load_telecom_activity_data():
#     # telecom_activity = gpd.read_feather('../Test Data/cleaned_3_days_grid_date.feather')
#     telecom_activity = gpd.read_feather('cleaned_3_days_grid_date.feather')
#     return telecom_activity
#
#
# aggregated_data = load_telecom_activity_data()


# def load_telecom_activity_data_time():
#     telecom_activity_time = gpd.read_feather('cleaned_all_days_grid.feather')
#     start_time = telecom_activity_time['start_time'].min()
#     start_time = start_time.value // 10 ** 6  # change datetime64[ns] to int(ms)
#     end_time = telecom_activity_time['start_time'].max()
#     end_time = end_time.value // 10 ** 6  # change datetime64[ns] to int(ms)
#     return telecom_activity_time, start_time, end_time

def load_telecom_interactions_data_date():
    # telecom_interactions_date = dd.read_parquet('interactions_parquet_data')
    telecom_interactions_date = vaex.open('interactions_parquet_data/MItoMI_all.hdf5')
    start_date_interactions = pd.Timestamp('2013-11-01')
    end_date_interactions = pd.Timestamp('2014-01-01')
    start_date_interactions = start_date_interactions.value // 10 ** 6
    end_date_interactions = end_date_interactions.value // 10 ** 6

    # start_date_interactions = start_date_interactions.to_datetime64()
    # end_date_interactions = end_date_interactions.to_datetime64()

    return telecom_interactions_date, start_date_interactions, end_date_interactions


telecom_interactions_date, start_date_interactions, end_date_interactions = load_telecom_interactions_data_date()

# print('telecom_interactions_date is:')
# print(len(telecom_interactions_date))
# print(telecom_interactions_date)


def load_telecom_activity_data_time():
    telecom_activity_time = dd.read_parquet('final_parquet_data')
    telecom_activity_time = telecom_activity_time.fillna(0)

    # Get the min and max start times
    start_time = telecom_activity_time['start_time'].min().compute()
    start_time = start_time.value // 10 ** 6  # change datetime64[ns] to int(ms)

    end_time = telecom_activity_time['start_time'].max().compute()
    end_time = end_time.value // 10 ** 6  # change datetime64[ns] to int(ms)

    return telecom_activity_time, start_time, end_time


aggregated_data_time, start_time, end_time = load_telecom_activity_data_time()
# print('telecom activity is:')
# print(len(aggregated_data_time))
# print(aggregated_data_time.head(10))



def load_telecom_activity_data_hour():
    telecom_activity_hour = dd.read_parquet('final_parquet_hour_data.parquet')

    # Get the min and max start times
    start_time_hour = telecom_activity_hour['start_time'].min().compute()
    start_time_hour = start_time_hour.value // 10 ** 6  # change datetime64[ns] to int(ms)

    end_time_hour = telecom_activity_hour['start_time'].max().compute()
    end_time_hour = end_time_hour.value // 10 ** 6  # change datetime64[ns] to int(ms)

    return telecom_activity_hour, start_time_hour, end_time_hour


aggregated_data_hour, start_time_hour, end_time_hour = load_telecom_activity_data_hour()
# print('telecom activity hour is:')
# print(len(aggregated_data_hour))
# print(aggregated_data_hour)


def load_telecom_activity_data_date():
    telecom_activity_date = pd.read_parquet('final_parquet_date_data.parquet')

    start_time_date = telecom_activity_date['start_time'].min()
    start_time_date = start_time_date.value // 10 ** 6

    end_time_date = telecom_activity_date['start_time'].max()
    end_time_date = end_time_date.value // 10 ** 6

    return telecom_activity_date, start_time_date, end_time_date


aggregated_data_date, start_time_date, end_time_date = load_telecom_activity_data_date()


weather_data = gpd.read_parquet('weather_data.parquet')
# print('weather_data is:')
# print(len(weather_data))
# print(weather_data.info())
# print(weather_data)


def load_news_data():
    news_data = gpd.read_parquet('final_milano_news_data.parquet')

    return news_data


news_data = load_news_data()

# print('news data is:')
# print(len(news_data))
# print(news_data.info())
# print(news_data)

activity_weather = gpd.read_file("activity_weather.geojson")

# print('activity_weather is:')
# print(len(activity_weather))
# print(activity_weather.info())
# print(activity_weather)

############################## precalculate cache ################################
# start_date = "2013-11-01"
# end_date = "2014-01-01"
#
# date_range = pd.date_range(start=start_date, end=end_date, freq='D')
# timestamps = (date_range.astype(int) / 10**6).tolist()
# all_time_periods = [[timestamps[i], timestamps[j]] for i in range(len(timestamps)) for j in range(i+1, len(timestamps))]


# print('start and end: ', start_time_date, end_time_date)

# aggregated_data = aggregated_data_time.copy()
# aggregated_data['start_time'] = pd.to_datetime(aggregated_data['start_time']).dt.date
#
# aggregated_data = aggregated_data_time.groupby(['name', 'start_time']).agg({
#     'sms_in': 'sum',
#     'sms_out': 'sum',
#     'call_in': 'sum',
#     'call_out': 'sum',
#     'internet': 'sum',
#     'geometry': 'first',
#     'cartodb_id': 'first'
# }).reset_index()
#
# aggregated_data = gpd.GeoDataFrame(aggregated_data, geometry='geometry')
