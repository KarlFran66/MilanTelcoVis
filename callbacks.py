import time
import geopandas as gpd
import json
import dash
import pandas as pd
from shapely.ops import unary_union
from dash.dependencies import Input, Output, State
from dash import callback_context
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import MultiPoint, Point, Polygon, box
from datetime import datetime
import dask_geopandas as dgpd
from dask.diagnostics import ProgressBar
import vaex
import random
from dash_extensions.javascript import assign
import logging
import igraph as ig
import gravis as gv
import os
import holoviews as hv
from holoviews import opts, dim
from bokeh.models import Div
from bokeh.layouts import column
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.colors as mcolors
import seaborn as sns

from app import cache
from utils import get_info, interactions_get_info, int_to_date, choropleth_style_handle, cluster_style_handle, \
    int_to_date_v3, blend_colors, get_info_weather, generate_bivariate_color_matrix, interactions_get_info_clusters
from config import data_options, colorscale, aggregated_data_time, aggregated_data_date, milano_grid, x_coords, \
    y_coords, colorscale_interaction, merged_origin_df, news_data, aggregated_data_hour, \
    telecom_interactions_date, start_date_interactions, end_date_interactions, weather_data, colorscale_blue, \
    colorscale_red, activity_weather
from activity import activity_layout
from interactions import interactions_layout
from activity_new import activity_new_layout
from activity_v3 import activity_v3_layout
from activity_v4 import activity_v4_layout
from weather import weather_layout
from weather_v2 import weather_layout_v2
from docs import docs_layout


def register_callbacks(app):
    # *******************************  callbacks for navbar  **********************************#
    # *****************************  past versions are hidden  ********************************#
    @app.callback(Output('page-content', 'children'),
                  [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/interactions':
            return interactions_layout()
        # elif pathname == '/weather_v1':
        #     return weather_layout()
        elif pathname == '/weather_v2':
            return weather_layout_v2()
        # elif pathname == '/activity_v1':
        #     return activity_layout()
        # elif pathname == '/activity_v2':
        #     return activity_new_layout()
        # elif pathname == '/activity_v3':
        #     return activity_v3_layout()
        elif pathname == '/docs':
            return docs_layout()
        else:
            return activity_v4_layout()

    # ***************************  callbacks for activity v4  ********************************#
    # **** try to go back to the full resolution, with more interactive range slider  ********#
    @app.callback(
        Output('feature-store-v4', 'data'),
        [Input('btn-sms-in-v4', 'n_clicks'),
         Input('btn-sms-out-v4', 'n_clicks'),
         Input('btn-call-in-v4', 'n_clicks'),
         Input('btn-call-out-v4', 'n_clicks'),
         Input('btn-internet-v4', 'n_clicks')]
    )
    def update_feature_store_v4(btn_sms_in, btn_sms_out, btn_call_in, btn_call_out, btn_internet):
        ctx = dash.callback_context

        selected_data = 'sms_in'

        if not ctx.triggered:
            selected_data = 'sms_in'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'btn-sms-in-v4':
                selected_data = 'sms_in'
            elif button_id == 'btn-sms-out-v4':
                selected_data = 'sms_out'
            elif button_id == 'btn-call-in-v4':
                selected_data = 'call_in'
            elif button_id == 'btn-call-out-v4':
                selected_data = 'call_out'
            elif button_id == 'btn-internet-v4':
                selected_data = 'internet'
        return {'feature': selected_data}

    @cache.memoize(timeout=None)
    def filtered_by_time_v4(low, high, selected_resolution):
        """
        Aggregate by square_id and merge with geo according to time slider
        :param low: start time
        :param high: end time
        :return: geo_df
        """
        # print('start filtering')
        print('filtered_by_time does not find cache')
        print(low, high)

        hour_or_date = None
        if selected_resolution == 'hour':
            hour_or_date = aggregated_data_hour
        elif selected_resolution == 'minute':
            hour_or_date = aggregated_data_time

        filtered_aggregated_data = hour_or_date[
            (hour_or_date['start_time'] >= low) & (hour_or_date['start_time'] <= high)]
        # print('finish filtering')
        agg_operations = {
            'sms_in': 'sum',
            'sms_out': 'sum',
            'call_in': 'sum',
            'call_out': 'sum',
            'internet': 'sum'
        }
        filtered_aggregated_data = filtered_aggregated_data.groupby('square_id').agg(agg_operations, split_out=5)
        filtered_aggregated_data = filtered_aggregated_data.compute().reset_index()
        filtered_aggregated_data = milano_grid.merge(filtered_aggregated_data, left_on='cellId', right_on='square_id')
        filtered_aggregated_data = filtered_aggregated_data.drop(columns=['cellId'])
        return filtered_aggregated_data

    @cache.memoize(timeout=None)  # cache for 60s
    def get_filtered_aggregated_data_v4(low, high, selected_area_gdf, flag, selected_resolution):
        """
        Recalculate the above geo_df if area is selected
        :param low: start time
        :param high: end time
        :param selected_area_gdf: selected area geo
        :param flag: True if area is selected
        :return: geo_df
        """
        print('get_filtered_aggregated_data does not find cache')
        filtered_aggregated_data = filtered_by_time_v4(low, high, selected_resolution)

        # if no selected area, then directly return
        if not flag:
            return filtered_aggregated_data

        # change selected_area_gdf CRS to milano_grid CRS
        selected_area_gdf.crs = "EPSG:4326"

        # join selected area and filtered aggregated data by intersects
        sjoined = gpd.sjoin(filtered_aggregated_data, selected_area_gdf, how='left', predicate='intersects')

        # create a column 'contained', set True if contained, otherwise False
        sjoined['contained'] = sjoined.index_right.notna()
        sjoined = sjoined.drop(
            columns=sjoined.columns.difference(
                ['geometry', 'contained'] + filtered_aggregated_data.columns.tolist()))
        sjoined = sjoined[~sjoined.index.duplicated(keep='first')]
        sjoined.reset_index(drop=True, inplace=True)

        # combine geometries of all contained grids
        contained_true_geometries = sjoined[sjoined['contained']]['geometry']
        combined_geometry = unary_union(contained_true_geometries)

        # replace contained grids' geometry with the combined geometry
        sjoined.loc[sjoined['contained'], 'geometry'] = combined_geometry

        aggregated = sjoined.groupby(sjoined['geometry']).agg(
            {
                'sms_in': 'sum',
                'sms_out': 'sum',
                'call_in': 'sum',
                'call_out': 'sum',
                'internet': 'sum',
                'square_id': lambda x: 'combined grids' if len(x) > 1 else x.iloc[0]
            }
        ).reset_index()

        aggregated = gpd.GeoDataFrame(aggregated, geometry='geometry')
        return aggregated

    ############################## precalculate cache ################################
    # @app.callback(
    #     Output("date-range-slider-v4", "value"),
    #     Input("interval-v4", "n_intervals")
    # )
    # def update_date_range_slider_v4(n):
    #     print('Mission: ', n)
    #     return all_time_periods[n]

    @app.callback(
        [Output("grid-v4", "data"),
         Output("grid-v4", "hideout"),
         Output("colorbar-container-v4", "children")],
        [Input('feature-store-v4', 'data'),
         Input("date-range-slider-v4", "value"),
         Input("start-date-time-slider-v4", "value"),
         Input("end-date-time-slider-v4", "value"),
         Input("radioitems-filter-v4", "value"),
         Input("edit-control-v4", "geojson")]
    )
    @cache.memoize(timeout=None)
    def update_map_new_v4(feature_data, slider_range, start_date_time, end_date_time, selected_resolution,
                          selected_area):
        print('update_map_new does not find cache')
        selected_data = feature_data['feature']

        # filtered by date-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        start_date_time_delta = pd.to_timedelta(start_date_time * 10, unit='min')
        end_date_time_delta = pd.to_timedelta(end_date_time * 10, unit='min')

        low = low + start_date_time_delta
        high = high + end_date_time_delta

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        aggregated = get_filtered_aggregated_data_v4(low, high, selected_area_gdf, flag, selected_resolution)

        # prepare classes and colorscale
        max_value = aggregated[selected_data].max()
        # print('filtered max_value: ', max_value)
        step = max_value / (len(colorscale) - 1)
        classes = [i * step for i in range(len(colorscale))]
        style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)

        # hideout dict
        hideout = dict(colorscale=colorscale, classes=classes, style=style, colorProp=selected_data)

        # categories and children for colorbar
        # ctg = ["{}+".format(int(cls)) for cls in classes[:-1]] + ["{}+".format(int(classes[-1]))]
        ctg = [""] * len(classes)
        colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=900, height=30,
                                            position="topright", style={'zIndex': 1000})

        return aggregated.__geo_interface__, hideout, colorbar

    # callback for info board
    @app.callback(
        Output("info-v4", "children"),
        [Input('feature-store-v4', 'data'),
         Input("grid-v4", "hoverData")]
    )
    def update_info_new_v4(feature_data, hover_data):
        selected_data = feature_data['feature']

        # info board
        selected_data_label = next((option['label'] for option in data_options if option['value'] == selected_data),
                                   None)
        info_children = get_info(selected_data_label, selected_data, hover_data)
        return info_children

    @app.callback(
        Output('offcanvas-v4', 'is_open'),
        [Input("grid-v4", "clickData"),
         Input("grid-v4", "n_clicks")],
        State('offcanvas-v4', 'is_open')
    )
    def toggle_offcanvas_v4(click_data, n_clicks, is_open):
        if click_data:
            is_open = True
        return is_open

    @cache.memoize(timeout=None)
    def get_filtered_df_v4(low, high, name, selected_area_gdf, flag, selected_resolution):
        hour_or_date = None
        if selected_resolution == 'hour':
            hour_or_date = aggregated_data_hour
        elif selected_resolution == 'minute':
            hour_or_date = aggregated_data_time

        if flag and name == 'combined grids':
            # print('start filtering')
            filtered_aggregated_data = hour_or_date[
                (hour_or_date['start_time'] >= low) &
                (hour_or_date['start_time'] <= high)
                ]
            # print('finish filtering')

            # change selected_area_gdf CRS to milano_grid CRS
            selected_area_gdf.crs = "EPSG:4326"

            sjoined = gpd.sjoin(milano_grid, selected_area_gdf, how='left', predicate='intersects')
            sjoined = sjoined.dropna(subset=['index_right'])
            covered_grids = sjoined['cellId'].unique().tolist()
            sjoined = filtered_aggregated_data[filtered_aggregated_data['square_id'].isin(covered_grids)]
            filtered_df = sjoined.groupby(sjoined['start_time']).agg(
                {
                    'sms_in': 'sum',
                    'sms_out': 'sum',
                    'call_in': 'sum',
                    'call_out': 'sum',
                    'internet': 'sum',
                }
            ).reset_index()
            filtered_df = filtered_df.compute()

            return filtered_df

        else:
            filtered_aggregated_data_time = hour_or_date[
                (hour_or_date['start_time'] >= low) &
                (hour_or_date['start_time'] <= high) &
                (hour_or_date['square_id'] == name)
                ]

            filtered_df = filtered_aggregated_data_time.compute()
            return filtered_df

    # callback for updating line plot
    @app.callback(
        Output("time-series-chart-v4", "figure"),
        [Input("grid-v4", "clickData"),
         Input("date-range-slider-v4", "value"),
         Input("start-date-time-slider-v4", "value"),
         Input("end-date-time-slider-v4", "value"),
         Input("checklist-v4", "value"),
         Input("radioitems-v4", "value"),
         Input("edit-control-v4", "geojson")]
    )
    @cache.memoize(timeout=None)
    def display_time_series_v4(clickData, slider_range, start_date_time, end_date_time, selected_types,
                               selected_resolution, selected_area):
        if clickData is None:
            raise PreventUpdate

        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        start_date_time_delta = pd.to_timedelta(start_date_time * 10, unit='min')
        end_date_time_delta = pd.to_timedelta(end_date_time * 10, unit='min')

        low = low + start_date_time_delta
        high = high + end_date_time_delta

        # name of the clicked neighbourhood
        name = clickData["properties"]["square_id"]

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        filtered_df = get_filtered_df_v4(low, high, name, selected_area_gdf, flag, selected_resolution)

        # filtered by checklist
        traces = []
        for col in selected_types:
            traces.append({
                "x": filtered_df["start_time"],
                "y": filtered_df[col],
                "type": "line",
                "name": next((option['label'] for option in data_options if option['value'] == col),
                             None)
            })

        # line plot
        figure = {
            "data": traces,
            "layout": {
                "title": {
                    "text": f"Telecom Activity Value over Time for Grid {name}",
                    "font": {"size": 25}
                },
                "xaxis": {
                    "title": {
                        "text": "Time",
                        "font": {"size": 20}
                    },
                    "tickfont": {"size": 20}
                },
                "yaxis": {
                    "title": {
                        "text": "Value",
                        "font": {"size": 20}
                    },
                    "tickfont": {"size": 20}
                },
                "hoverlabel": {
                    "font": {"size": 18}
                },
                "legend": {
                    "font": {"size": 18}
                }
            }
        }

        return figure

    @cache.memoize(timeout=None)
    def generate_geojson_news_markers_with_popup_v4(news_gdf):
        features = []
        for _, row in news_gdf.iterrows():
            feature = {
                "type": "Feature",
                "geometry": row['geometry'].__geo_interface__,
                "properties": {
                    "popup": f"<strong style='font-size: 40px;'>News Title: </strong>{row['title']}<br>" 
                             f"<a href='{row['link']}' target='_blank' style='font-size: 40px;'>Link</a><br>" 
                             f"<strong style='font-size: 40px;'>Publish Time: </strong>{row['timestamp']}<br>" 
                             f"<strong style='font-size: 40px;'>Topic: </strong>{row['topic']}<br>" 
                             f"<strong style='font-size: 40px;'>Address: </strong>{row['address']}"
                }
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    @app.callback(
        Output("news-markers-v4", "data"),
        Input("date-range-slider-v4", "value")
    )
    @cache.memoize(timeout=None)
    def display_news_markers_v4(slider_range):
        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]
        filtered_news_data = news_data[
            (news_data['timestamp'] >= low) &
            (news_data['timestamp'] <= high)
            ]
        news_geojson_with_popup = generate_geojson_news_markers_with_popup_v4(filtered_news_data)
        return news_geojson_with_popup

    # ************************************  callbacks for interaction v1 test  *****************************#
    # callback for info board
    @app.callback(
        Output("interactions-info", "children"),
        [Input("interactions-grid", "hoverData"),
         Input('interactions-type-store', 'data')]
    )
    def interactions_update_info_new(hover_data, type_data):
        # info board
        if type_data['type'] == 'btn_choropleth':
            info_children = interactions_get_info(hover_data)
            return info_children
        else:
            info_children = interactions_get_info_clusters(hover_data)
            return info_children

    @app.callback(
        Output('interactions-type-store', 'data'),
        [Input('btn-choropleth', 'n_clicks'),
         Input('btn-clustering', 'n_clicks')]
    )
    def update_feature_store_interactions(btn_choropleth, btn_clustering):
        ctx = dash.callback_context

        selected_type = 'btn_choropleth'

        if not ctx.triggered:
            selected_type = 'btn_choropleth'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'btn-choropleth':
                selected_type = 'btn_choropleth'
            elif button_id == 'btn-clustering':
                selected_type = 'btn_clustering'
        return {'type': selected_type}

    # @cache.memoize(timeout=None)
    def filter_by_time_interactions_new(low, high):
        # vaex.settings.default_thread_pool = True
        # start_all = time.time()
        # start_time = time.time()
        # print(f"Start filtering by date period from {low} to {high}.")
        # filtered_aggregated_data = telecom_interactions_date[
        #     (telecom_interactions_date['start_time'] >= low) & (telecom_interactions_date['start_time'] <= high)]
        # # print("The dataframe now becomes:")
        # # print(filtered_aggregated_data)
        # end_time = time.time()
        # print("Finished filtering, time spent: ", end_time - start_time)
        #
        # print(" ")
        # print("Start groupby...")
        # start_time = time.time()
        # filtered_aggregated_data = filtered_aggregated_data.groupby(by=['node1', 'node2'],
        #                                                             agg={'DIS': vaex.agg.sum('DIS')})
        # filtered_aggregated_data['node1'] = filtered_aggregated_data['node1'].astype('int16')
        # filtered_aggregated_data['node2'] = filtered_aggregated_data['node2'].astype('int16')
        # filtered_aggregated_data['DIS'] = filtered_aggregated_data['DIS'].astype('float32')
        # end_time = time.time()
        # print('End groupby, time spent: ', end_time - start_time)
        #
        # print("Transform to pandas dataframe...")
        # start_time = time.time()
        # filtered_aggregated_data = filtered_aggregated_data.to_pandas_df()
        # end_time = time.time()
        # print('Transformed to df, time spent: ', end_time - start_time)
        #
        # print(filtered_aggregated_data)
        # end_all = time.time()
        # print('Total time spent: ', end_all - start_all)
        #
        # return filtered_aggregated_data

        # low_str = low.strftime('%Y-%m-%d')
        # high_str = high.strftime('%Y-%m-%d')

        filename = f'every_single_interactions/filtered_aggregated_data_{low}_to_{high}.parquet'

        if os.path.exists(filename):
            print(f"Reading precomputed data from {filename}.")
            filtered_aggregated_data = pd.read_parquet(filename)
        else:
            print(f"File {filename} does not exist.")
            filtered_aggregated_data = pd.DataFrame(columns=['node1', 'node2', 'DIS'])

        return filtered_aggregated_data

    @cache.memoize(timeout=None)
    def get_filtered_aggregated_data_interactions(low, high):
        filtered_aggregated_data = filter_by_time_interactions_new(low, high)

        print("Calculate choropleth map...")
        # find self loops and aggregate self-loop DIS for each node
        self_loops = filtered_aggregated_data[
            filtered_aggregated_data['node1'] == filtered_aggregated_data['node2']]
        self_loops_sum = self_loops.groupby('node1')['DIS'].sum()

        # calculate DIS sums for node1 and node2 except self loops
        weights_node1 = filtered_aggregated_data.groupby('node1')['DIS'].sum()
        weights_node2 = filtered_aggregated_data.groupby('node2')['DIS'].sum()

        # add DIS sums for node1 and node2
        node_weights = weights_node1.add(weights_node2, fill_value=0)

        # subtract self-loop DIS
        node_weights = node_weights.sub(self_loops_sum, fill_value=0).reset_index()

        node_weights.columns = ['node', 'DIS']
        node_weights = milano_grid.merge(node_weights, left_on='cellId', right_on='node')
        node_weights = node_weights.drop(columns=['cellId'])
        print(f"End processing within {low} and {high}.")

        return node_weights

    @cache.memoize(timeout=None)
    def get_clustered_graph_new9_test(low, high, resolution):
        filtered_aggregated_data = filter_by_time_interactions_new(low, high)

        # change node id to 0-9999
        filtered_aggregated_data['node1'] -= 1
        filtered_aggregated_data['node2'] -= 1

        # build undirected graph
        edges = list(zip(filtered_aggregated_data['node1'], filtered_aggregated_data['node2']))
        weights = filtered_aggregated_data['DIS'].tolist()
        print(' ')
        print('start building graph...')
        start_time = time.time()
        G = ig.Graph(edges=edges, directed=False)
        G.es['weight'] = weights
        end_time = time.time()
        print('end building graph, time spent: ', end_time - start_time)

        print('number of nodes: ', G.vcount())

        print(' ')
        print('apply leiden algorithm...')
        start_time = time.time()
        if resolution == '1':
            partition = G.community_leiden(objective_function='modularity', weights='weight', resolution=1)
        elif resolution == '2':
            partition = G.community_leiden(objective_function='modularity', weights='weight', resolution=2)
        end_time = time.time()
        print('finish, time spent: ', end_time - start_time)
        print(partition.sizes())

        palette = sns.color_palette('tab20', len(partition.sizes()))
        colors = [sns.color_palette(palette).as_hex()[i] for i in range(len(partition.sizes()))]
        # print(colors)
        # colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in partition.sizes()]

        print(' ')
        print('start aggregating by clusters...')
        cluster_labels = partition.membership
        G.contract_vertices(cluster_labels, combine_attrs="sum")
        G.simplify(multiple=True, loops=False, combine_edges=dict(weight="sum"))

        return G, partition, colors

    @app.callback(
        [Output("interactions-grid", "data"),
         Output("interactions-grid", "hideout"),
         Output("interactions-grid", "style"),
         Output("interactions-colorbar-container", "children")],
        [Input('interactions-type-store', 'data'),
         Input("date-range-slider-interactions", 'value'),
         Input("radioitems-cluster-resolution", "value")]
    )
    # @cache.memoize(timeout=None)
    def update_map_interactions(type_data, slider_range, resolution):
        low, high = slider_range
        low = int_to_date_v3(low)
        high = int_to_date_v3(high)

        if type_data['type'] == 'btn_choropleth':
            aggregated = get_filtered_aggregated_data_interactions(low, high)

            # prepare classes and colorscale
            max_value = aggregated['DIS'].max()
            step = max_value / (len(colorscale_interaction) - 1)
            classes = [i * step for i in range(len(colorscale_interaction))]
            style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)

            # hideout dict
            hideout = dict(colorscale=colorscale_interaction, classes=classes, style=style, colorProp='DIS')

            # categories and children for colorbar
            # ctg = ["{}+".format(int(cls)) for cls in classes[:-1]] + ["{}+".format(int(classes[-1]))]
            ctg = [""] * len(classes)
            colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale_interaction, width=900,
                                                height=30,
                                                position="topright", style={'zIndex': 1000})

            return aggregated.__geo_interface__, hideout, choropleth_style_handle, colorbar

        elif type_data['type'] == 'btn_clustering':
            print(' ')
            print('read from cache...')
            start_time = time.time()
            G, partition, colors = get_clustered_graph_new9_test(low, high, resolution)
            end_time = time.time()
            print('finish, time spent: ', end_time - start_time)

            print(' ')
            print('build geodf...')
            start_time = time.time()
            cluster_labels = partition.membership
            nodes = range(1, 10001)
            cluster_df = pd.DataFrame({'node': nodes, 'cluster_label': cluster_labels})
            merged_cluster_df = milano_grid.merge(cluster_df, left_on='cellId', right_on='node', how='inner')
            merged_cluster_df = merged_cluster_df.drop(columns=['cellId'])
            merged_cluster_df = gpd.GeoDataFrame(merged_cluster_df, geometry='geometry')
            dissolved_merged_cluster_df = merged_cluster_df.dissolve(by='cluster_label')
            dissolved_merged_cluster_df = dissolved_merged_cluster_df.drop(columns=['node'])
            dissolved_merged_cluster_df = dissolved_merged_cluster_df.reset_index()
            end_time = time.time()
            print('finish, time spent: ', end_time - start_time)
            print(dissolved_merged_cluster_df)

            # clusters = dissolved_merged_cluster_df['cluster_label'].unique()

            # generate random color for each cluster
            # colors = {cluster: "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for cluster in
            #           clusters}

            # colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for cluster in
            #           clusters]
            style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)
            hideout = dict(colors=colors, style=style, length=len(colors))

            return dissolved_merged_cluster_df.__geo_interface__, hideout, cluster_style_handle, None

    @app.callback(
        Output('offcanvas-interactions', 'is_open'),
        [Input("interactions-grid", "clickData"),
         Input("interactions-grid", "n_clicks")],
        State('offcanvas-interactions', 'is_open')
    )
    def toggle_offcanvas_interactions(click_data, n_clicks, is_open):
        if click_data:
            is_open = True
        return is_open

    @app.callback(
        Output('interactions-map', 'style'),
        Input('offcanvas-interactions', 'is_open')
    )
    def update_map_style(is_open):
        if is_open:
            style = {"width": "60vw", "height": "95vh", "position": "absolute"}
        else:
            style = {"width": "100vw", "height": "95vh", "position": "absolute"}

        return style

    @app.callback(
        [Output("network-plot", "srcDoc"),
         Output("chord-diagram", "srcDoc")],
        [Input("interactions-grid", "clickData"),
         Input("date-range-slider-interactions", 'value'),
         Input("radioitems-cluster-resolution", "value"),
         Input("checklist-chord", "value"), ]
    )
    # @cache.memoize(timeout=None)
    def display_network_interactions(clickData, slider_range, resolution, self_interactions):
        if clickData is None:
            raise PreventUpdate

        low, high = slider_range
        low = int_to_date_v3(low)
        high = int_to_date_v3(high)

        print(' ')
        print('read from cache...')
        start_time = time.time()
        G, partition, colors = get_clustered_graph_new9_test(low, high, resolution)
        end_time = time.time()
        print('finish, time spent: ', end_time - start_time)

        # print(' ')
        # print('start aggregating by clusters...')
        # cluster_labels = partition.membership
        # G.contract_vertices(cluster_labels, combine_attrs="sum")
        # G.simplify(multiple=True, loops=False, combine_edges=dict(weight="sum"))

        degree_list = G.strength(weights='weight')
        max_degree = max(degree_list)
        # normalize and scale node size
        G.vs['size'] = [20 * (degree + 1) / (max_degree + 1) for degree in degree_list]
        # G.vs['color'] = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        #                  for i in range(len(G.vs))]
        G.vs['color'] = colors
        max_weight = max(G.es['weight'])
        # normalize and scale edge width
        G.es['size'] = [5 * weight / max_weight for weight in G.es['weight']]

        fig = gv.d3(G, graph_height=500, details_height=100, edge_curvature=0.3)

        if os.path.exists('network_graph.html'):
            os.remove('network_graph.html')

        fig.export_html('network_graph.html')

        network_fig = open('network_graph.html', 'r').read()

        print('plot chord diagram')
        print('change to df')
        cluster_edges = [(edge.source, edge.target, edge["weight"]) for edge in G.es]
        df_cluster_graph = pd.DataFrame(cluster_edges, columns=['source', 'target', 'value'])
        # df_cluster_graph['source'] = df_cluster_graph['source'].astype(str)
        # df_cluster_graph['target'] = df_cluster_graph['target'].astype(str)
        df_cluster_graph['label'] = 'cluster ' + df_cluster_graph['source'].astype(str)
        print(df_cluster_graph)

        if len(self_interactions) == 0:
            df_cluster_graph = df_cluster_graph[df_cluster_graph['source'] != df_cluster_graph['target']]

        hv.extension('bokeh')
        chord = hv.Chord(df_cluster_graph).opts(
            opts.Chord(
                cmap=colors,
                edge_cmap=colors,
                edge_color='source',
                labels='label',
                node_color='source',
                width=500,
                height=500
            )
        )

        if os.path.exists('chord_diagram.html'):
            os.remove('chord_diagram.html')
        hv.save(chord, 'chord_diagram.html', fmt='html')
        chord_fig = open('chord_diagram.html', 'r').read()

        return network_fig, chord_fig

    @cache.memoize(timeout=None)
    def generate_geojson_news_markers_with_popup_interactions(news_gdf):
        features = []
        for _, row in news_gdf.iterrows():
            feature = {
                "type": "Feature",
                "geometry": row['geometry'].__geo_interface__,
                "properties": {
                    "popup": f"<strong>News Title: </strong>{row['title']}<br>"
                             f"<a href='{row['link']}' target='_blank'>Link</a><br>"
                             f"<strong>Publish Time: </strong>{row['timestamp']}<br>"
                             f"<strong>Topic: </strong>{row['topic']}<br>"
                             f"<strong>Address: </strong>{row['address']}"
                }
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    @app.callback(
        Output("news-markers-interactions", "data"),
        Input("date-range-slider-interactions", "value")
    )
    @cache.memoize(timeout=None)
    def display_news_markers_interactions(slider_range):
        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]
        filtered_news_data = news_data[
            (news_data['timestamp'] >= low) &
            (news_data['timestamp'] <= high)
            ]
        news_geojson_with_popup = generate_geojson_news_markers_with_popup_v4(filtered_news_data)
        return news_geojson_with_popup

    #  ************************  callbacks for weather v2  ***************************#
    @app.callback(
        [Output("bivariate-grid-v2", "data"),
         Output("bivariate-grid-v2", "hideout"),
         Output('bivariate-colorscale-container', 'children')],
        [Input("radioitems-weather-v2-activity", "value"),
         Input("radioitems-weather-v2-weather", "value")]
    )
    def update_map_weather_v2(activity_feature, weather_feature):
        telecom_colorscale = colorscale_red
        precipitation_colorscale = colorscale_blue

        # calculate colorscale
        max_value = activity_weather[activity_feature].max()
        min_value = activity_weather[activity_feature].min()
        step = (max_value - min_value) / len(telecom_colorscale)
        classes = [min_value + i * step for i in range(len(telecom_colorscale))]

        max_value_precipitation = activity_weather[weather_feature].max()
        min_value_precipitation = activity_weather[weather_feature].min()
        step_precipitation = (max_value_precipitation - min_value_precipitation) / len(precipitation_colorscale)
        classes_precipitation = [min_value_precipitation + i * step_precipitation for i in
                                 range(len(precipitation_colorscale))]

        bivariate_colorscale = [[ci, pi] for ci in telecom_colorscale for pi in precipitation_colorscale]
        blended_bivariate_colorscale = [blend_colors(pair[0], pair[1]) for pair in bivariate_colorscale]

        hideout = dict(
            activity_feature=activity_feature,
            weather_feature=weather_feature,
            blended_bivariate_colorscale=blended_bivariate_colorscale,
            classes=classes,
            classes_precipitation=classes_precipitation,
            precipitation_colorscale=precipitation_colorscale,
            style=dict(weight=0, opacity=1, color='white', fillOpacity=0.7)
        )

        return activity_weather.__geo_interface__, hideout, generate_bivariate_color_matrix(
            blended_bivariate_colorscale, 5, 'Telecom', 'Weather')

    @app.callback(
        [Output("grid-weather-v2", "data"),
         Output("grid-weather-v2", "hideout")],
        [Input("radioitems-weather-v2-weather", "value")]
    )
    def update_weather_map_weather_v2(weather_feature):
        precipitation_colorscale = colorscale_blue
        max_value_precipitation = activity_weather[weather_feature].max()
        min_value_precipitation = activity_weather[weather_feature].min()
        step_precipitation = (max_value_precipitation - min_value_precipitation) / len(precipitation_colorscale)
        classes_precipitation = [min_value_precipitation + i * step_precipitation for i in
                                 range(len(precipitation_colorscale))]
        style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)
        hideout = dict(colorscale=precipitation_colorscale, classes=classes_precipitation, style=style,
                       colorProp=weather_feature)

        return activity_weather.__geo_interface__, hideout

    @app.callback(
        [Output("grid-activity-v2", "data"),
         Output("grid-activity-v2", "hideout")],
        [Input("radioitems-weather-v2-activity", "value")]
    )
    def update_weather_map_weather_v2(activity_feature):
        telecom_colorscale = colorscale_red
        max_value = activity_weather[activity_feature].max()
        min_value = activity_weather[activity_feature].min()
        step = (max_value - min_value) / len(telecom_colorscale)
        classes = [min_value + i * step for i in range(len(telecom_colorscale))]
        style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)
        hideout = dict(colorscale=telecom_colorscale, classes=classes, style=style, colorProp=activity_feature)

        return activity_weather.__geo_interface__, hideout

    @app.callback(
        Output('sensor-markers', 'children'),
        [Input("radioitems-weather-v2-weather", "value")]
    )
    def generate_sensor_markers_v2(weather_feature):
        precipitation_df = weather_data[weather_data['type'] == weather_feature]
        agg_operations = {
            'street_name': 'first',
            'type': 'first',
            'UOM': 'first',
            'geometry': 'first',
            'value': 'mean'
        }
        precipitation_df = precipitation_df.groupby('sensor_id').agg(agg_operations).reset_index()

        markers = [
            dl.Marker(
                position=[row['geometry'].y, row['geometry'].x],
                children=[
                    dl.Tooltip(f"Sensor {row['sensor_id']}, Average {row['type']}: {row['value']:.2f} {row['UOM']}",
                               className='dl-tooltip')
                ]
            ) for index, row in precipitation_df.iterrows()
        ]
        return markers

    @app.callback(
        Output("info-weather-v2", "children"),
        [Input("radioitems-weather-v2-activity", "value"),
         Input("radioitems-weather-v2-weather", "value"),
         Input("bivariate-grid-v2", "hoverData")]
    )
    def update_info_weather_v2(activity_feature, weather_feature, hover_data):
        info_children = get_info_weather(activity_feature, weather_feature, hover_data)
        return info_children

    @app.callback(
        [Output("parallel-coordinates-v2", "figure"),
         Output("scatter-v2", "figure"),
         Output("line-v2", "figure")],
        [Input("radioitems-weather-v2-activity", "value"),
         Input("radioitems-weather-v2-weather", "value"),
         Input("radioitems-select-days-v2", "value"),
         Input("checklist-log-v2", "value"),
         Input("radioitems-select-sensor-v2", "value")]
    )
    def generate_parallel_coordinates(activity_feature, weather_feature, selected_days, selected_log,
                                      selected_sensor):
        precipitation_date_df = weather_data[weather_data['type'] == weather_feature].copy()
        precipitation_date_df['date'] = precipitation_date_df['start_time'].dt.normalize()
        agg_operations = {
            'street_name': 'first',
            'type': 'first',
            'UOM': 'first',
            'geometry': 'first',
            'value': 'mean'
        }
        precipitation_date_df = precipitation_date_df.groupby(['date', 'sensor_id']).agg(
            agg_operations).reset_index()

        precipitation_df = weather_data[weather_data['type'] == weather_feature]
        agg_operations = {
            'street_name': 'first',
            'type': 'first',
            'UOM': 'first',
            'geometry': 'first',
            'value': 'mean'
        }
        precipitation_df = precipitation_df.groupby('sensor_id').agg(agg_operations).reset_index()

        precipitation_df = gpd.GeoDataFrame(precipitation_df, geometry='geometry')
        grid_df = milano_grid.set_crs('EPSG:4326', allow_override=True)
        precipitation_df = precipitation_df.set_crs('EPSG:4326', allow_override=True)
        result = gpd.sjoin(grid_df, precipitation_df, how="inner", predicate='contains')

        activity_date_df = aggregated_data_date.copy()
        needed_square_ids = result['cellId'].unique()
        filtered_activity_df = activity_date_df[activity_date_df['square_id'].isin(needed_square_ids)]

        grid_to_sensor_map = dict(zip(result['cellId'], result['sensor_id']))
        filtered_activity_df_copy = filtered_activity_df.copy()
        filtered_activity_df_copy['sensor_id'] = filtered_activity_df['square_id'].map(grid_to_sensor_map)
        final_merged_df = pd.merge(
            filtered_activity_df_copy,
            precipitation_date_df,
            left_on=['sensor_id', 'start_time'],
            right_on=['sensor_id', 'date']
        )

        plot_df = final_merged_df.copy()
        # plot_df = final_merged_df[final_merged_df['square_id'] == square_id]
        if 'Log Scale Activity' in selected_log:
            plot_df[activity_feature] = np.log1p(plot_df[activity_feature])
        if 'Log Scale Weather' in selected_log:
            plot_df['value'] = np.log1p(plot_df['value'])
        if selected_days == 'Only Weekdays':
            plot_df = plot_df[plot_df['date'].dt.dayofweek < 5]
        if selected_days == 'Only Weekends':
            plot_df = plot_df[plot_df['date'].dt.dayofweek >= 5]
        if 'Filter Zeros' in selected_log:
            plot_df = plot_df[plot_df['value'] > 0.1]

        fig = px.parallel_coordinates(
            plot_df,
            dimensions=[activity_feature, 'value'],
            color='value',
            labels={
                "sms_in": "SMS In",
                "sms_out": "SMS Out",
                "call_in": " Call In",
                "call_out": "Call Out",
                "internet": "Internet",
                "value": "Weather Feature"
            },
            color_continuous_scale=px.colors.diverging.Tealrose,
            title=f"Parallel Coordinates Plot"
        )

        fig.update_layout(
            title={'font': {'size': 18}},
            legend={'font': {'size': 18}},
            font={'size': 18},
            autosize=True
        )

        fig_scatter = px.scatter(
            plot_df,
            x=activity_feature,
            y='value',
            # color='date',
            labels={
                "sms_in": "SMS In",
                "sms_out": "SMS Out",
                "call_in": " Call In",
                "call_out": "Call Out",
                "internet": "Internet",
                "value": "Weather Feature Value"
            },
            title="Scatter Plot of Telecom Activity vs. Weather Feature",
            trendline="ols",
            marginal_x="histogram",
            marginal_y="histogram",
            template="simple_white",
            hover_data={'date': True, 'sensor_id': True}
        )

        fig_scatter.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig_scatter.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        fig_scatter.update_traces(
            hovertemplate=(
                    'Internet: %{x:.2f}<br>' +
                    'Weather Feature Value: %{y:.2f}<br>' +
                    'Date: %{customdata[0]}<br>' +
                    'Sensor ID: %{customdata[1]}'
            ),
            customdata=plot_df[['date', 'sensor_id']].applymap(
                lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x).values
        )

        fig_scatter.update_layout(
            autosize=True,
            title={'font': {'size': 20}},
            xaxis={'title': {'font': {'size': 18}}, 'tickfont': {'size': 16}},
            yaxis={'title': {'font': {'size': 18}}, 'tickfont': {'size': 16}},
            legend={'font': {'size': 18}},
            hoverlabel={'font': {'size': 16}}
        )

        sensor_id = selected_sensor
        plot_df = final_merged_df[final_merged_df['sensor_id'] == sensor_id]

        fig_line = go.Figure()
        # fig_line.add_trace(go.Scatter(x=plot_df['start_time'], y=plot_df['value'], name=weather_feature, mode='lines',
        #                          line=dict(color='blue')))
        # fig_line.add_trace(go.Scatter(x=plot_df['start_time'], y=plot_df[activity_feature], name=activity_feature, mode='lines',
        #                          line=dict(color='red'), yaxis='y2'))

        fig_line.add_trace(go.Scatter(
            x=plot_df['start_time'],
            y=plot_df['value'],
            name=weather_feature,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=5)
        ))

        fig_line.add_trace(go.Scatter(
            x=plot_df['start_time'],
            y=plot_df[activity_feature],
            name=activity_feature,
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=5),
            yaxis='y2'
        ))

        fig_line.update_layout(
            title={'text': 'Weather Feature and Telecom Activity Over Time', 'font': {'size': 20}},
            xaxis=dict(
                title={'text': 'Time', 'font': {'size': 18}},
                tickfont={'size': 16}
            ),
            yaxis=dict(
                title={'text': 'Weather Feature', 'font': {'size': 18}},
                tickfont={'size': 16},
                side='left',
                showgrid=False,
            ),
            yaxis2=dict(
                title={'text': 'Telecom Activity', 'font': {'size': 18}},
                tickfont={'size': 16},
                side='right',
                overlaying='y',
                showgrid=False
            ),
            legend={'font': {'size': 18}},
            hoverlabel={'font': {'size': 16}}
        )

        return fig, fig_scatter, fig_line

    @app.callback(
        [Output("radioitems-select-sensor-v2", "options"),
         Output("radioitems-select-sensor-v2", "value")],
        [Input("radioitems-weather-v2-weather", "value")]
    )
    def generate_parallel_coordinates(weather_feature):
        precipitation_df = weather_data[weather_data['type'] == weather_feature]
        unique_sensor_ids = precipitation_df['sensor_id'].unique()
        options = [{'label': 'Sensor ' + sensor_id.astype(str), 'value': sensor_id} for sensor_id in
                   unique_sensor_ids]
        value = unique_sensor_ids[1]
        return options, value

    # ************************ the rest are all past versions ******************************#

    # ************************  callbacks for activity v2 **********************************#
    # **********************  v2 beautifies layout with bootstrap  *************************#
    @app.callback(
        Output('feature-store', 'data'),
        [Input('btn-sms-in', 'n_clicks'),
         Input('btn-sms-out', 'n_clicks'),
         Input('btn-call-in', 'n_clicks'),
         Input('btn-call-out', 'n_clicks'),
         Input('btn-internet', 'n_clicks')]
    )
    def update_feature_store(btn_sms_in, btn_sms_out, btn_call_in, btn_call_out, btn_internet):
        ctx = dash.callback_context

        selected_data = 'sms_in'

        if not ctx.triggered:
            selected_data = 'sms_in'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'btn-sms-in':
                selected_data = 'sms_in'
            elif button_id == 'btn-sms-out':
                selected_data = 'sms_out'
            elif button_id == 'btn-call-in':
                selected_data = 'call_in'
            elif button_id == 'btn-call-out':
                selected_data = 'call_out'
            elif button_id == 'btn-internet':
                selected_data = 'internet'
        return {'feature': selected_data}

    @cache.memoize(timeout=None)
    def filtered_by_time(low, high):
        """
        Aggregate by square_id and merge with geo according to time slider
        :param low: start time
        :param high: end time
        :return: geo_df
        """
        # print('start filtering')
        # print('filtered_by_time does not find cache')
        # print(low, high)
        filtered_aggregated_data = aggregated_data_time[
            (aggregated_data_time['start_time'] >= low) & (aggregated_data_time['start_time'] <= high)]
        # print('finish filtering')
        agg_operations = {
            'sms_in': 'sum',
            'sms_out': 'sum',
            'call_in': 'sum',
            'call_out': 'sum',
            'internet': 'sum'
        }
        filtered_aggregated_data = filtered_aggregated_data.groupby('square_id').agg(agg_operations)
        filtered_aggregated_data = filtered_aggregated_data.compute().reset_index()
        filtered_aggregated_data = milano_grid.merge(filtered_aggregated_data, left_on='cellId', right_on='square_id')
        filtered_aggregated_data = filtered_aggregated_data.drop(columns=['cellId'])
        return filtered_aggregated_data

    @cache.memoize(timeout=None)  # cache for 60s
    def get_filtered_aggregated_data(low, high, selected_area_gdf, flag):
        """
        Recalculate the above geo_df if area is selected
        :param low: start time
        :param high: end time
        :param selected_area_gdf: selected area geo
        :param flag: True if area is selected
        :return: geo_df
        """
        # print('get_filtered_aggregated_data does not find cache')
        filtered_aggregated_data = filtered_by_time(low, high)

        # if no selected area, then directly return
        if not flag:
            return filtered_aggregated_data

        # change selected_area_gdf CRS to milano_grid CRS
        selected_area_gdf.crs = "EPSG:4326"

        # join selected area and filtered aggregated data by intersects
        sjoined = gpd.sjoin(filtered_aggregated_data, selected_area_gdf, how='left', predicate='intersects')

        # create a column 'contained', set True if contained, otherwise False
        sjoined['contained'] = sjoined.index_right.notna()
        sjoined = sjoined.drop(
            columns=sjoined.columns.difference(['geometry', 'contained'] + filtered_aggregated_data.columns.tolist()))
        sjoined = sjoined[~sjoined.index.duplicated(keep='first')]
        sjoined.reset_index(drop=True, inplace=True)

        # combine geometries of all contained grids
        contained_true_geometries = sjoined[sjoined['contained']]['geometry']
        combined_geometry = unary_union(contained_true_geometries)

        # replace contained grids' geometry with the combined geometry
        sjoined.loc[sjoined['contained'], 'geometry'] = combined_geometry

        aggregated = sjoined.groupby(sjoined['geometry']).agg(
            {
                'sms_in': 'sum',
                'sms_out': 'sum',
                'call_in': 'sum',
                'call_out': 'sum',
                'internet': 'sum',
                'square_id': lambda x: 'combined grids' if len(x) > 1 else x.iloc[0]
            }
        ).reset_index()

        aggregated = gpd.GeoDataFrame(aggregated, geometry='geometry')
        return aggregated

    # call back for updating map when data type or slider changes
    # @app.callback(
    #     [Output("grid", "data"),
    #      Output("grid", "hideout"),
    #      Output("colorbar-container", "children")],
    #     [Input("data-selector", "value"),
    #      Input("date-range-slider", "value"),
    #      Input("edit-control", "geojson")]
    # )
    # @cache.memoize(timeout=None)
    # def update_map(selected_data, slider_range, selected_area):
    #     # filtered by date-range-slider
    #     low, high = slider_range
    #     low = int_to_date(low)  # change int to datetime64[ns]
    #     high = int_to_date(high)  # change int to datetime64[ns]
    #
    #     flag = True
    #
    #     # aggregate selected area
    #     if selected_area is not None and 'features' in selected_area and selected_area['features']:
    #         # If an area is selected, grab geojson info of the selected area
    #         selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
    #         selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
    #     else:
    #         # Otherwise, create a null geodf
    #         selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
    #         flag = False
    #
    #     aggregated = get_filtered_aggregated_data(low, high, selected_area_gdf, flag)
    #
    #     # prepare classes and colorscale
    #     max_value = aggregated[selected_data].max()
    #     # print('filtered max_value: ', max_value)
    #     step = max_value / (len(colorscale) - 1)
    #     classes = [i * step for i in range(len(colorscale))]
    #     style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)
    #
    #     # hideout dict
    #     hideout = dict(colorscale=colorscale, classes=classes, style=style, colorProp=selected_data)
    #
    #     # categories and children for colorbar
    #     ctg = ["{}+".format(int(cls)) for cls in classes[:-1]] + ["{}+".format(int(classes[-1]))]
    #     colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=900, height=30,
    #                                         position="bottomleft")
    #
    #     return aggregated.__geo_interface__, hideout, colorbar

    @app.callback(
        [Output("grid", "data"),
         Output("grid", "hideout"),
         Output("colorbar-container", "children")],
        [Input('feature-store', 'data'),
         Input("date-range-slider", "value"),
         Input("edit-control", "geojson")]
    )
    @cache.memoize(timeout=None)
    def update_map_new(feature_data, slider_range, selected_area):
        # print('update_map_new does not find cache')
        selected_data = feature_data['feature']

        # filtered by date-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        aggregated = get_filtered_aggregated_data(low, high, selected_area_gdf, flag)

        # prepare classes and colorscale
        max_value = aggregated[selected_data].max()
        # print('filtered max_value: ', max_value)
        step = max_value / (len(colorscale) - 1)
        classes = [i * step for i in range(len(colorscale))]
        style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)

        # hideout dict
        hideout = dict(colorscale=colorscale, classes=classes, style=style, colorProp=selected_data)

        # categories and children for colorbar
        ctg = ["{}+".format(int(cls)) for cls in classes[:-1]] + ["{}+".format(int(classes[-1]))]
        colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=900, height=30,
                                            position="topright")

        return aggregated.__geo_interface__, hideout, colorbar

    # callback for info board
    # @app.callback(
    #     Output("info", "children"),
    #     [Input("grid", "hoverData"),
    #      Input("data-selector", "value")]
    # )
    # def update_info(hover_data, selected_data):
    #     # info board
    #     selected_data_label = next((option['label'] for option in data_options if option['value'] == selected_data),
    #                                None)
    #     info_children = get_info(selected_data_label, selected_data, hover_data)
    #     return info_children

    # callback for info board
    @app.callback(
        Output("info", "children"),
        [Input('feature-store', 'data'),
         Input("grid", "hoverData")]
    )
    def update_info_new(feature_data, hover_data):
        selected_data = feature_data['feature']

        # info board
        selected_data_label = next((option['label'] for option in data_options if option['value'] == selected_data),
                                   None)
        info_children = get_info(selected_data_label, selected_data, hover_data)
        return info_children

    # callback for opening and closing the details window
    @app.callback(
        Output("popup", "style"),
        [Input("grid", "clickData"),
         Input("close", "n_clicks")],
        [State("popup", "style")]
    )
    def toggle_popup(clickData, n_clicks_close, style):
        ctx = callback_context

        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if trigger_id == "grid":
            # jumpbox slide in
            style["right"] = "0"
            return style
        elif trigger_id == "close":
            # jumpbox slide out
            style["right"] = "-100%"
            return style

        return style

    @app.callback(
        Output('offcanvas', 'is_open'),
        Input("grid", "clickData"),
        State('offcanvas', 'is_open')
    )
    def toggle_offcanvas(click_data, is_open):
        if click_data:
            is_open = True
            # return not is_open
        return is_open

    @cache.memoize(timeout=None)
    def get_filtered_df(low, high, name, selected_area_gdf, flag):
        if flag and name == 'combined grids':
            # print('start filtering')
            filtered_aggregated_data = aggregated_data_time[
                (aggregated_data_time['start_time'] >= low) &
                (aggregated_data_time['start_time'] <= high)
                ]
            # print('finish filtering')

            # change selected_area_gdf CRS to milano_grid CRS
            selected_area_gdf.crs = "EPSG:4326"

            sjoined = gpd.sjoin(milano_grid, selected_area_gdf, how='left', predicate='intersects')
            sjoined = sjoined.dropna(subset=['index_right'])
            covered_grids = sjoined['cellId'].unique().tolist()
            sjoined = filtered_aggregated_data[filtered_aggregated_data['square_id'].isin(covered_grids)]
            filtered_df = sjoined.groupby(sjoined['start_time']).agg(
                {
                    'sms_in': 'sum',
                    'sms_out': 'sum',
                    'call_in': 'sum',
                    'call_out': 'sum',
                    'internet': 'sum',
                }
            ).reset_index()
            filtered_df = filtered_df.compute()

            return filtered_df

        else:
            filtered_aggregated_data_time = aggregated_data_time[
                (aggregated_data_time['start_time'] >= low) &
                (aggregated_data_time['start_time'] <= high) &
                (aggregated_data_time['square_id'] == name)
                ]

            filtered_df = filtered_aggregated_data_time.compute()
            return filtered_df

    # callback for updating line plot
    @app.callback(
        Output("time-series-chart", "figure"),
        [Input("grid", "clickData"),
         Input("date-range-slider", "value"),
         Input("checklist", "value"),
         Input("edit-control", "geojson")]
    )
    @cache.memoize(timeout=None)
    def display_time_series(clickData, slider_range, selected_types, selected_area):
        if clickData is None:
            raise PreventUpdate

        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        # name of the clicked neighbourhood
        name = clickData["properties"]["square_id"]

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        filtered_df = get_filtered_df(low, high, name, selected_area_gdf, flag)

        # filtered by checklist
        traces = []
        for col in selected_types:
            traces.append({
                "x": filtered_df["start_time"],
                "y": filtered_df[col],
                "type": "line",
                "name": next((option['label'] for option in data_options if option['value'] == col),
                             None)
            })

        # line plot
        figure = {
            "data": traces,
            "layout": {
                "title": f"Value over Time for Grid: {name}",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Value"}
            }
        }

        return figure

    # @app.callback(
    #     Output("time-series-chart", "figure"),
    #     [Input("grid", "clickData"),
    #      Input("date-range-slider", "value"),
    #      Input("checklist", "value"),
    #      Input("edit-control", "geojson")]
    # )
    # @cache.memoize(timeout=None)
    # def update_time_series(clickData, slider_range, selected_types, selected_area):
    #     if clickData is None:
    #         raise PreventUpdate
    #
    #     # filtered by data-range-slider
    #     low, high = slider_range
    #     low = int_to_date(low)  # change int to datetime64[ns]
    #     high = int_to_date(high)  # change int to datetime64[ns]
    #
    #     # name of the clicked neighbourhood
    #     name = clickData["properties"]["square_id"]
    #
    #     flag = True
    #
    #     # aggregate selected area
    #     if selected_area is not None and 'features' in selected_area and selected_area['features']:
    #         # If an area is selected, grab geojson info of the selected area
    #         selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
    #         selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
    #     else:
    #         # Otherwise, create a null geodf
    #         selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
    #         flag = False
    #
    #     filtered_df = get_filtered_df(low, high, name, selected_area_gdf, flag)
    #
    #     # filtered by checklist
    #     traces = []
    #     for col in selected_types:
    #         traces.append({
    #             "x": filtered_df["start_time"],
    #             "y": filtered_df[col],
    #             "type": "line",
    #             "name": next((option['label'] for option in data_options if option['value'] == col),
    #                          None)
    #         })
    #
    #     # line plot
    #     figure = {
    #         "data": traces,
    #         "layout": {
    #             "title": f"Value over Time for Grid: {name}",
    #             "xaxis": {"title": "Time"},
    #             "yaxis": {"title": "Value"}
    #         }
    #     }
    #
    #     return figure

    #  **********************  callbacks for activity v3  *************************#
    #  ******* v3 reduce resolution, add news data, layer control  ****************#
    @app.callback(
        Output('feature-store-v3', 'data'),
        [Input('btn-sms-in-v3', 'n_clicks'),
         Input('btn-sms-out-v3', 'n_clicks'),
         Input('btn-call-in-v3', 'n_clicks'),
         Input('btn-call-out-v3', 'n_clicks'),
         Input('btn-internet-v3', 'n_clicks')]
    )
    def update_feature_store_v3(btn_sms_in, btn_sms_out, btn_call_in, btn_call_out, btn_internet):
        ctx = dash.callback_context

        selected_data = 'sms_in'

        if not ctx.triggered:
            selected_data = 'sms_in'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'btn-sms-in-v3':
                selected_data = 'sms_in'
            elif button_id == 'btn-sms-out-v3':
                selected_data = 'sms_out'
            elif button_id == 'btn-call-in-v3':
                selected_data = 'call_in'
            elif button_id == 'btn-call-out-v3':
                selected_data = 'call_out'
            elif button_id == 'btn-internet-v3':
                selected_data = 'internet'
        return {'feature': selected_data}

    @cache.memoize(timeout=None)
    def filtered_by_time_v3(low, high):
        """
        Aggregate by square_id and merge with geo according to time slider
        :param low: start time
        :param high: end time
        :return: geo_df
        """
        print('filtered_by_time does not find cache')
        print(low, high)
        filtered_aggregated_data = aggregated_data_date[
            (aggregated_data_date['start_time'] >= low) & (aggregated_data_date['start_time'] <= high)]
        agg_operations = {
            'sms_in': 'sum',
            'sms_out': 'sum',
            'call_in': 'sum',
            'call_out': 'sum',
            'internet': 'sum'
        }
        filtered_aggregated_data = filtered_aggregated_data.groupby('square_id').agg(agg_operations).reset_index()
        filtered_aggregated_data = milano_grid.merge(filtered_aggregated_data, left_on='cellId', right_on='square_id')
        filtered_aggregated_data = filtered_aggregated_data.drop(columns=['cellId'])

        return filtered_aggregated_data

    @cache.memoize(timeout=None)  # cache for 60s
    def get_filtered_aggregated_data_v3(low, high, selected_area_gdf, flag):
        """
        Recalculate the above geo_df if area is selected
        :param low: start time
        :param high: end time
        :param selected_area_gdf: selected area geo
        :param flag: True if area is selected
        :return: geo_df
        """
        print('get_filtered_aggregated_data does not find cache')
        filtered_aggregated_data = filtered_by_time_v3(low, high)

        # if no selected area, then directly return
        if not flag:
            return filtered_aggregated_data

        # change selected_area_gdf CRS to milano_grid CRS
        selected_area_gdf.crs = "EPSG:4326"

        # join selected area and filtered aggregated data by intersects
        sjoined = gpd.sjoin(filtered_aggregated_data, selected_area_gdf, how='left', predicate='intersects')

        # create a column 'contained', set True if contained, otherwise False
        sjoined['contained'] = sjoined.index_right.notna()
        sjoined = sjoined.drop(
            columns=sjoined.columns.difference(
                ['geometry', 'contained'] + filtered_aggregated_data.columns.tolist()))
        sjoined = sjoined[~sjoined.index.duplicated(keep='first')]
        sjoined.reset_index(drop=True, inplace=True)

        # combine geometries of all contained grids
        contained_true_geometries = sjoined[sjoined['contained']]['geometry']
        combined_geometry = unary_union(contained_true_geometries)

        # replace contained grids' geometry with the combined geometry
        sjoined.loc[sjoined['contained'], 'geometry'] = combined_geometry

        aggregated = sjoined.groupby(sjoined['geometry']).agg(
            {
                'sms_in': 'sum',
                'sms_out': 'sum',
                'call_in': 'sum',
                'call_out': 'sum',
                'internet': 'sum',
                'square_id': lambda x: 'combined grids' if len(x) > 1 else x.iloc[0]
            }
        ).reset_index()

        aggregated = gpd.GeoDataFrame(aggregated, geometry='geometry')
        return aggregated

    @app.callback(
        [Output("grid-v3", "data"),
         Output("grid-v3", "hideout"),
         Output("colorbar-container-v3", "children")],
        [Input('feature-store-v3', 'data'),
         Input("date-range-slider-v3", "value"),
         Input("edit-control-v3", "geojson")]
    )
    @cache.memoize(timeout=None)
    def update_map_new_v3(feature_data, slider_range, selected_area):
        print('update_map_new does not find cache')
        selected_data = feature_data['feature']

        # filtered by date-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        aggregated = get_filtered_aggregated_data_v3(low, high, selected_area_gdf, flag)

        # prepare classes and colorscale
        max_value = aggregated[selected_data].max()
        # print('filtered max_value: ', max_value)
        step = max_value / (len(colorscale) - 1)
        classes = [i * step for i in range(len(colorscale))]
        style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)

        # hideout dict
        hideout = dict(colorscale=colorscale, classes=classes, style=style, colorProp=selected_data)

        # categories and children for colorbar
        # ctg = ["{}+".format(int(cls)) for cls in classes[:-1]] + ["{}+".format(int(classes[-1]))]
        ctg = [""] * len(classes)
        colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=900, height=30,
                                            position="topright")

        return aggregated.__geo_interface__, hideout, colorbar

    # callback for info board
    @app.callback(
        Output("info-v3", "children"),
        [Input('feature-store-v3', 'data'),
         Input("grid-v3", "hoverData")]
    )
    def update_info_new_v3(feature_data, hover_data):
        selected_data = feature_data['feature']

        # info board
        selected_data_label = next((option['label'] for option in data_options if option['value'] == selected_data),
                                   None)
        info_children = get_info(selected_data_label, selected_data, hover_data)
        return info_children

    @app.callback(
        Output('offcanvas-v3', 'is_open'),
        [Input("grid-v3", "clickData"),
         Input("grid-v3", "n_clicks")],
        State('offcanvas-v3', 'is_open')
    )
    def toggle_offcanvas_v3(click_data, n_clicks, is_open):
        if click_data:
            is_open = True
        return is_open

    @cache.memoize(timeout=None)
    def get_filtered_df_v3(low, high, name, selected_area_gdf, flag):
        if flag and name == 'combined grids':
            # print('start filtering')
            filtered_aggregated_data = aggregated_data_time[
                (aggregated_data_time['start_time'] >= low) &
                (aggregated_data_time['start_time'] <= high)
                ]
            # print('finish filtering')

            # change selected_area_gdf CRS to milano_grid CRS
            selected_area_gdf.crs = "EPSG:4326"

            sjoined = gpd.sjoin(milano_grid, selected_area_gdf, how='left', predicate='intersects')
            sjoined = sjoined.dropna(subset=['index_right'])
            covered_grids = sjoined['cellId'].unique().tolist()
            sjoined = filtered_aggregated_data[filtered_aggregated_data['square_id'].isin(covered_grids)]
            filtered_df = sjoined.groupby(sjoined['start_time']).agg(
                {
                    'sms_in': 'sum',
                    'sms_out': 'sum',
                    'call_in': 'sum',
                    'call_out': 'sum',
                    'internet': 'sum',
                }
            ).reset_index()
            filtered_df = filtered_df.compute()

            return filtered_df

        else:
            filtered_aggregated_data_time = aggregated_data_time[
                (aggregated_data_time['start_time'] >= low) &
                (aggregated_data_time['start_time'] <= high) &
                (aggregated_data_time['square_id'] == name)
                ]

            filtered_df = filtered_aggregated_data_time.compute()
            return filtered_df

    # callback for updating line plot
    @app.callback(
        Output("time-series-chart-v3", "figure"),
        [Input("grid-v3", "clickData"),
         Input("date-range-slider-v3", "value"),
         Input("checklist-v3", "value"),
         Input("edit-control-v3", "geojson")]
    )
    @cache.memoize(timeout=None)
    def display_time_series_v3(clickData, slider_range, selected_types, selected_area):
        if clickData is None:
            raise PreventUpdate

        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        # name of the clicked neighbourhood
        name = clickData["properties"]["square_id"]

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        filtered_df = get_filtered_df_v3(low, high, name, selected_area_gdf, flag)

        # filtered by checklist
        traces = []
        for col in selected_types:
            traces.append({
                "x": filtered_df["start_time"],
                "y": filtered_df[col],
                "type": "line",
                "name": next((option['label'] for option in data_options if option['value'] == col),
                             None)
            })

        # line plot
        figure = {
            "data": traces,
            "layout": {
                "title": f"Value over Time for Grid: {name}",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Value"}
            }
        }

        return figure

    @cache.memoize(timeout=None)
    def generate_geojson_news_markers_with_popup_v3(news_gdf):
        features = []
        for _, row in news_gdf.iterrows():
            feature = {
                "type": "Feature",
                "geometry": row['geometry'].__geo_interface__,
                "properties": {
                    "popup": f"<strong>News Title: </strong>{row['title']}<br>"
                             f"<a href='{row['link']}' target='_blank'>Link</a><br>"
                             f"<strong>Publish Time: </strong>{row['timestamp']}<br>"
                             f"<strong>Topic: </strong>{row['topic']}<br>"
                             f"<strong>Address: </strong>{row['address']}"
                }
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    @app.callback(
        Output("news-markers-v3", "data"),
        Input("date-range-slider-v3", "value")
    )
    @cache.memoize(timeout=None)
    def display_news_markers_v3(slider_range):
        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]
        filtered_news_data = news_data[
            (news_data['timestamp'] >= low) &
            (news_data['timestamp'] <= high)
            ]
        news_geojson_with_popup = generate_geojson_news_markers_with_popup_v3(filtered_news_data)
        return news_geojson_with_popup

    # **************************  callbacks for weather v1  ********************* #
    @app.callback(
        Output('feature-store-weather', 'data'),
        [Input('btn-sms-in-weather', 'n_clicks'),
         Input('btn-sms-out-weather', 'n_clicks'),
         Input('btn-call-in-weather', 'n_clicks'),
         Input('btn-call-out-weather', 'n_clicks'),
         Input('btn-internet-weather', 'n_clicks')]
    )
    def update_feature_store_weather(btn_sms_in, btn_sms_out, btn_call_in, btn_call_out, btn_internet):
        ctx = dash.callback_context

        selected_data = 'sms_in'

        if not ctx.triggered:
            selected_data = 'sms_in'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'btn-sms-in-weather':
                selected_data = 'sms_in'
            elif button_id == 'btn-sms-out-weather':
                selected_data = 'sms_out'
            elif button_id == 'btn-call-in-weather':
                selected_data = 'call_in'
            elif button_id == 'btn-call-out-weather':
                selected_data = 'call_out'
            elif button_id == 'btn-internet-weather':
                selected_data = 'internet'
        return {'feature': selected_data}

    # @cache.memoize(timeout=None)
    def weather_filtered_by_time_weather(low, high, selected_resolution):
        """
        Aggregate by square_id and merge with geo according to time slider
        :param low: start time
        :param high: end time
        :return: geo_df
        """
        # print('start filtering')
        print('weather_filtered_by_time_weather does not find cache')
        print(low, high)

        # hour_or_date = None
        # if selected_resolution == 'hour':
        #     hour_or_date = aggregated_data_hour
        # elif selected_resolution == 'minute':
        #     hour_or_date = aggregated_data_time

        filtered_aggregated_data = weather_data[
            (weather_data['start_time'] >= low) & (weather_data['start_time'] <= high) &
            (weather_data['type'] == 'Precipitation')]
        # print('finish filtering')
        agg_operations = {
            'street_name': 'first',
            'type': 'first',
            'UOM': 'first',
            'geometry': 'first',
            'value': 'sum'
        }
        filtered_aggregated_data = filtered_aggregated_data.groupby('sensor_id').agg(agg_operations).reset_index()
        print(filtered_aggregated_data)

        return filtered_aggregated_data

    @cache.memoize(timeout=None)
    def get_filtered_aggregated_weather_data_weather(low, high, selected_area_gdf, flag, selected_resolution):
        print('get_filtered_aggregated_data does not find cache')
        filtered_aggregated_data = weather_filtered_by_time_weather(low, high, selected_resolution)

        coordinates_list = [(point.x, point.y) for point in filtered_aggregated_data.geometry]
        coordinates = np.array(coordinates_list)
        values = filtered_aggregated_data['value'].values
        OK = OrdinaryKriging(
            coordinates[:, 0], coordinates[:, 1], values,
            variogram_model='spherical', verbose=False, enable_plotting=False
        )
        z, ss = OK.execute('grid', x_coords, y_coords)

        temp = get_filtered_aggregated_data_weather(low, high, selected_area_gdf, flag, selected_resolution)
        temp['precipitation'] = z.data.flatten()

        # if no selected area, then directly return
        if not flag:
            return temp

        # change selected_area_gdf CRS to milano_grid CRS
        selected_area_gdf.crs = "EPSG:4326"

        # join selected area and filtered aggregated data by intersects
        sjoined = gpd.sjoin(filtered_aggregated_data, selected_area_gdf, how='left', predicate='intersects')

        # create a column 'contained', set True if contained, otherwise False
        sjoined['contained'] = sjoined.index_right.notna()
        sjoined = sjoined.drop(
            columns=sjoined.columns.difference(
                ['geometry', 'contained'] + filtered_aggregated_data.columns.tolist()))
        sjoined = sjoined[~sjoined.index.duplicated(keep='first')]
        sjoined.reset_index(drop=True, inplace=True)

        # combine geometries of all contained grids
        contained_true_geometries = sjoined[sjoined['contained']]['geometry']
        combined_geometry = unary_union(contained_true_geometries)

        # replace contained grids' geometry with the combined geometry
        sjoined.loc[sjoined['contained'], 'geometry'] = combined_geometry

        aggregated = sjoined.groupby(sjoined['geometry']).agg(
            {
                'sms_in': 'sum',
                'sms_out': 'sum',
                'call_in': 'sum',
                'call_out': 'sum',
                'internet': 'sum',
                'square_id': lambda x: 'combined grids' if len(x) > 1 else x.iloc[0]
            }
        ).reset_index()

        aggregated = gpd.GeoDataFrame(aggregated, geometry='geometry')
        return aggregated

    @app.callback(
        [Output("weather-grid-weather", "data"),
         Output("weather-grid-weather", "hideout"),
         Output("weather-colorbar-container-weather", "children")],
        [Input("date-range-slider-weather", "value"),
         Input("start-date-time-slider-weather", "value"),
         Input("end-date-time-slider-weather", "value"),
         Input("radioitems-filter-weather", "value"),
         Input("edit-control-weather", "geojson")]
    )
    # @cache.memoize(timeout=None)
    def update_weather_map_new_weather(slider_range, start_date_time, end_date_time, selected_resolution,
                                       selected_area):
        print('update_weather_map_new_weather does not find cache')

        # filtered by date-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        start_date_time_delta = pd.to_timedelta(start_date_time * 10, unit='min')
        end_date_time_delta = pd.to_timedelta(end_date_time * 10, unit='min')

        low = low + start_date_time_delta
        high = high + end_date_time_delta

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        aggregated = get_filtered_aggregated_weather_data_weather(low, high, selected_area_gdf, flag,
                                                                  selected_resolution)

        print('final weather dataset:')
        print(aggregated)

        colorscale_blue = [
            '#f7fbff', '#ecf4fc', '#e2eef8', '#d8e7f5', '#cee0f2', '#c2d9ee',
            '#b1d2e8', '#a0cbe2', '#8bc0dd', '#76b4d8', '#62a8d3', '#519ccc',
            '#4090c5', '#3282be', '#2474b7', '#1967ad', '#0f59a3', '#084c94',
            '#083e80', '#08306b'
        ]

        max_value = aggregated['precipitation'].max()
        min_value = aggregated['precipitation'].min()
        step = (max_value - min_value) / len(colorscale_blue)
        classes = [min_value + i * step for i in range(len(colorscale_blue))]

        print(max_value, min_value)
        print(classes)

        style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)

        # ctg = ["{}+".format(cls) for cls in classes]
        ctg = [""] * len(classes)
        colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale_blue, width=900, height=30,
                                            position="bottomleft")
        hideout = dict(colorscale=colorscale_blue, classes=classes, style=style, colorProp='precipitation')

        print(aggregated)

        return aggregated.__geo_interface__, hideout, colorbar

    @app.callback(
        [Output("bivariate-grid", "data"),
         Output("bivariate-grid", "hideout")],
        [Input("date-range-slider-weather", "value"),
         Input("start-date-time-slider-weather", "value"),
         Input("end-date-time-slider-weather", "value"),
         Input("radioitems-filter-weather", "value"),
         Input("edit-control-weather", "geojson")]
    )
    # @cache.memoize(timeout=None)
    def update_bivariate_map(slider_range, start_date_time, end_date_time, selected_resolution,
                             selected_area):
        print('update_bivariate_map does not find cache')

        # filtered by date-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        start_date_time_delta = pd.to_timedelta(start_date_time * 10, unit='min')
        end_date_time_delta = pd.to_timedelta(end_date_time * 10, unit='min')

        low = low + start_date_time_delta
        high = high + end_date_time_delta

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        df_merged = get_filtered_aggregated_weather_data_weather(low, high, selected_area_gdf, flag,
                                                                 selected_resolution)

        telecom_colorscale = [
            '#fff5f0', '#ffece3', '#fee3d7', '#fdd6c5', '#fdc7b0', '#fcb79c',
            '#fca588', '#fc9474', '#fc8363', '#fb7252', '#f86044', '#f34c37',
            '#ed392b', '#de2a25', '#cf1c1f', '#bf151a', '#af1117', '#9b0d14',
            '#810610', '#67000d'
        ]
        precipitation_colorscale = [
            '#f7fbff', '#ecf4fc', '#e2eef8', '#d8e7f5', '#cee0f2', '#c2d9ee',
            '#b1d2e8', '#a0cbe2', '#8bc0dd', '#76b4d8', '#62a8d3', '#519ccc',
            '#4090c5', '#3282be', '#2474b7', '#1967ad', '#0f59a3', '#084c94',
            '#083e80', '#08306b'
        ]

        # calculate colorscale
        max_value = df_merged['call_out'].max()
        min_value = df_merged['call_out'].min()
        step = (max_value - min_value) / len(telecom_colorscale)
        classes = [min_value + i * step for i in range(len(telecom_colorscale))]

        max_value_precipitation = df_merged['precipitation'].max()
        min_value_precipitation = df_merged['precipitation'].min()
        step_precipitation = (max_value_precipitation - min_value_precipitation) / len(precipitation_colorscale)
        classes_precipitation = [min_value_precipitation + i * step_precipitation for i in
                                 range(len(precipitation_colorscale))]

        bivariate_colorscale = [[ci, pi] for ci in telecom_colorscale for pi in precipitation_colorscale]

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

        blended_bivariate_colorscale = [blend_colors(pair[0], pair[1]) for pair in bivariate_colorscale]

        hideout = dict(
            blended_bivariate_colorscale=blended_bivariate_colorscale,
            classes=classes,
            classes_precipitation=classes_precipitation,
            precipitation_colorscale=precipitation_colorscale,
            style=dict(weight=0, opacity=1, color='white', fillOpacity=0.7)
        )

        return df_merged.__geo_interface__, hideout

    # @cache.memoize(timeout=None)
    def filtered_by_time_weather(low, high, selected_resolution):
        """
        Aggregate by square_id and merge with geo according to time slider
        :param low: start time
        :param high: end time
        :return: geo_df
        """
        # print('start filtering')
        print('filtered_by_time does not find cache')
        print(low, high)

        hour_or_date = None
        if selected_resolution == 'hour':
            hour_or_date = aggregated_data_hour
        elif selected_resolution == 'minute':
            hour_or_date = aggregated_data_time

        filtered_aggregated_data = hour_or_date[
            (hour_or_date['start_time'] >= low) & (hour_or_date['start_time'] <= high)]
        # print('finish filtering')
        agg_operations = {
            'sms_in': 'sum',
            'sms_out': 'sum',
            'call_in': 'sum',
            'call_out': 'sum',
            'internet': 'sum'
        }
        filtered_aggregated_data = filtered_aggregated_data.groupby('square_id').agg(agg_operations, split_out=5)
        filtered_aggregated_data = filtered_aggregated_data.compute().reset_index()
        filtered_aggregated_data = milano_grid.merge(filtered_aggregated_data, left_on='cellId',
                                                     right_on='square_id')
        filtered_aggregated_data = filtered_aggregated_data.drop(columns=['cellId'])
        return filtered_aggregated_data

    @cache.memoize(timeout=None)
    def get_filtered_aggregated_data_weather(low, high, selected_area_gdf, flag, selected_resolution):
        """
        Recalculate the above geo_df if area is selected
        :param low: start time
        :param high: end time
        :param selected_area_gdf: selected area geo
        :param flag: True if area is selected
        :return: geo_df
        """
        print('get_filtered_aggregated_data does not find cache')
        filtered_aggregated_data = filtered_by_time_weather(low, high, selected_resolution)

        # if no selected area, then directly return
        if not flag:
            return filtered_aggregated_data

        # change selected_area_gdf CRS to milano_grid CRS
        selected_area_gdf.crs = "EPSG:4326"

        # join selected area and filtered aggregated data by intersects
        sjoined = gpd.sjoin(filtered_aggregated_data, selected_area_gdf, how='left', predicate='intersects')

        # create a column 'contained', set True if contained, otherwise False
        sjoined['contained'] = sjoined.index_right.notna()
        sjoined = sjoined.drop(
            columns=sjoined.columns.difference(
                ['geometry', 'contained'] + filtered_aggregated_data.columns.tolist()))
        sjoined = sjoined[~sjoined.index.duplicated(keep='first')]
        sjoined.reset_index(drop=True, inplace=True)

        # combine geometries of all contained grids
        contained_true_geometries = sjoined[sjoined['contained']]['geometry']
        combined_geometry = unary_union(contained_true_geometries)

        # replace contained grids' geometry with the combined geometry
        sjoined.loc[sjoined['contained'], 'geometry'] = combined_geometry

        aggregated = sjoined.groupby(sjoined['geometry']).agg(
            {
                'sms_in': 'sum',
                'sms_out': 'sum',
                'call_in': 'sum',
                'call_out': 'sum',
                'internet': 'sum',
                'square_id': lambda x: 'combined grids' if len(x) > 1 else x.iloc[0]
            }
        ).reset_index()

        aggregated = gpd.GeoDataFrame(aggregated, geometry='geometry')
        return aggregated

    ############################## precalculate cache ################################
    # @app.callback(
    #     Output("date-range-slider-weather", "value"),
    #     Input("interval-weather", "n_intervals")
    # )
    # def update_date_range_slider_weather(n):
    #     print('Mission: ', n)
    #     return all_time_periods[n]

    @app.callback(
        [Output("grid-weather", "data"),
         Output("grid-weather", "hideout"),
         Output("colorbar-container-weather", "children")],
        [Input('feature-store-weather', 'data'),
         Input("date-range-slider-weather", "value"),
         Input("start-date-time-slider-weather", "value"),
         Input("end-date-time-slider-weather", "value"),
         Input("radioitems-filter-weather", "value"),
         Input("edit-control-weather", "geojson")]
    )
    # @cache.memoize(timeout=None)
    def update_map_new_weather(feature_data, slider_range, start_date_time, end_date_time, selected_resolution,
                               selected_area):
        print('update_map_new does not find cache')
        selected_data = feature_data['feature']

        # filtered by date-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        start_date_time_delta = pd.to_timedelta(start_date_time * 10, unit='min')
        end_date_time_delta = pd.to_timedelta(end_date_time * 10, unit='min')

        low = low + start_date_time_delta
        high = high + end_date_time_delta

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        aggregated = get_filtered_aggregated_data_weather(low, high, selected_area_gdf, flag, selected_resolution)

        # prepare classes and colorscale
        max_value = aggregated[selected_data].max()
        # print('filtered max_value: ', max_value)

        colorscale_red = [
            '#fff5f0', '#ffece3', '#fee3d7', '#fdd6c5', '#fdc7b0', '#fcb79c',
            '#fca588', '#fc9474', '#fc8363', '#fb7252', '#f86044', '#f34c37',
            '#ed392b', '#de2a25', '#cf1c1f', '#bf151a', '#af1117', '#9b0d14',
            '#810610', '#67000d'
        ]

        step = max_value / (len(colorscale_red) - 1)
        classes = [i * step for i in range(len(colorscale_red))]
        style = dict(weight=0, opacity=30, color='white', fillOpacity=0.7)

        # hideout dict
        hideout = dict(colorscale=colorscale_red, classes=classes, style=style, colorProp=selected_data)

        # categories and children for colorbar
        # ctg = ["{}+".format(int(cls)) for cls in classes[:-1]] + ["{}+".format(int(classes[-1]))]
        ctg = [""] * len(classes)
        colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale_red, width=900, height=30,
                                            position="topright", style={'zIndex': 1000})

        return aggregated.__geo_interface__, hideout, colorbar

    # callback for info board
    @app.callback(
        Output("info-weather", "children"),
        [Input('feature-store-weather', 'data'),
         Input("grid-weather", "hoverData")]
    )
    def update_info_new_weather(feature_data, hover_data):
        selected_data = feature_data['feature']

        # info board
        selected_data_label = next((option['label'] for option in data_options if option['value'] == selected_data),
                                   None)
        info_children = get_info(selected_data_label, selected_data, hover_data)
        return info_children

    @app.callback(
        Output('offcanvas-weather', 'is_open'),
        [Input("grid-weather", "clickData"),
         Input("grid-weather", "n_clicks")],
        State('offcanvas-weather', 'is_open')
    )
    def toggle_offcanvas_weather(click_data, n_clicks, is_open):
        if click_data:
            is_open = True
        return is_open

    # @cache.memoize(timeout=None)
    def get_filtered_df_weather(low, high, name, selected_area_gdf, flag, selected_resolution):
        hour_or_date = None
        if selected_resolution == 'hour':
            hour_or_date = aggregated_data_hour
        elif selected_resolution == 'minute':
            hour_or_date = aggregated_data_time

        if flag and name == 'combined grids':
            # print('start filtering')
            filtered_aggregated_data = hour_or_date[
                (hour_or_date['start_time'] >= low) &
                (hour_or_date['start_time'] <= high)
                ]
            # print('finish filtering')

            # change selected_area_gdf CRS to milano_grid CRS
            selected_area_gdf.crs = "EPSG:4326"

            sjoined = gpd.sjoin(milano_grid, selected_area_gdf, how='left', predicate='intersects')
            sjoined = sjoined.dropna(subset=['index_right'])
            covered_grids = sjoined['cellId'].unique().tolist()
            sjoined = filtered_aggregated_data[filtered_aggregated_data['square_id'].isin(covered_grids)]
            filtered_df = sjoined.groupby(sjoined['start_time']).agg(
                {
                    'sms_in': 'sum',
                    'sms_out': 'sum',
                    'call_in': 'sum',
                    'call_out': 'sum',
                    'internet': 'sum',
                }
            ).reset_index()
            filtered_df = filtered_df.compute()

            return filtered_df

        else:
            filtered_aggregated_data_time = hour_or_date[
                (hour_or_date['start_time'] >= low) &
                (hour_or_date['start_time'] <= high) &
                (hour_or_date['square_id'] == name)
                ]

            filtered_df = filtered_aggregated_data_time.compute()
            return filtered_df

    # callback for updating line plot
    @app.callback(
        Output("time-series-chart-weather", "figure"),
        [Input("grid-weather", "clickData"),
         Input("date-range-slider-weather", "value"),
         Input("start-date-time-slider-weather", "value"),
         Input("end-date-time-slider-weather", "value"),
         Input("checklist-weather", "value"),
         Input("radioitems-weather", "value"),
         Input("edit-control-weather", "geojson")]
    )
    # @cache.memoize(timeout=None)
    def display_time_series_weather(clickData, slider_range, start_date_time, end_date_time, selected_types,
                                    selected_resolution, selected_area):
        if clickData is None:
            raise PreventUpdate

        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]

        start_date_time_delta = pd.to_timedelta(start_date_time * 10, unit='min')
        end_date_time_delta = pd.to_timedelta(end_date_time * 10, unit='min')

        low = low + start_date_time_delta
        high = high + end_date_time_delta

        # name of the clicked neighbourhood
        name = clickData["properties"]["square_id"]

        flag = True

        # aggregate selected area
        if selected_area is not None and 'features' in selected_area and selected_area['features']:
            # If an area is selected, grab geojson info of the selected area
            selected_area_gdf = gpd.GeoDataFrame.from_features(selected_area['features'])
            selected_area_gdf = selected_area_gdf.drop(columns=['_leaflet_id', 'type', '_bounds'], errors='ignore')
        else:
            # Otherwise, create a null geodf
            selected_area_gdf = gpd.GeoDataFrame(columns=['geometry'])
            flag = False

        filtered_df = get_filtered_df_v4(low, high, name, selected_area_gdf, flag, selected_resolution)

        # filtered by checklist
        traces = []
        for col in selected_types:
            traces.append({
                "x": filtered_df["start_time"],
                "y": filtered_df[col],
                "type": "line",
                "name": next((option['label'] for option in data_options if option['value'] == col),
                             None)
            })

        # line plot
        figure = {
            "data": traces,
            "layout": {
                "title": f"Value over Time for Grid: {name}",
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Value"}
            }
        }

        return figure

    # @cache.memoize(timeout=None)
    def generate_geojson_news_markers_with_popup_weather(news_gdf):
        features = []
        for _, row in news_gdf.iterrows():
            feature = {
                "type": "Feature",
                "geometry": row['geometry'].__geo_interface__,
                "properties": {
                    "popup": f"<strong>News Title: </strong>{row['title']}<br>"
                             f"<a href='{row['link']}' target='_blank'>Link</a><br>"
                             f"<strong>Publish Time: </strong>{row['timestamp']}<br>"
                             f"<strong>Topic: </strong>{row['topic']}<br>"
                             f"<strong>Address: </strong>{row['address']}"
                }
            }
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    @app.callback(
        Output("news-markers-weather", "data"),
        Input("date-range-slider-weather", "value")
    )
    # @cache.memoize(timeout=None)
    def display_news_markers_weather(slider_range):
        # filtered by data-range-slider
        low, high = slider_range
        low = int_to_date(low)  # change int to datetime64[ns]
        high = int_to_date(high)  # change int to datetime64[ns]
        filtered_news_data = news_data[
            (news_data['timestamp'] >= low) &
            (news_data['timestamp'] <= high)
            ]
        news_geojson_with_popup = generate_geojson_news_markers_with_popup_v4(filtered_news_data)
        return news_geojson_with_popup
