from app import app
from layouts import global_layout
from callbacks import register_callbacks
from dask.distributed import Client, LocalCluster
from dask.distributed import Client, LocalCluster
import logging
from dash.dependencies import Input, Output, State


# def cluster():
#     logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

#     logging.info("Starting LocalCluster")
#     try:

#         cluster = LocalCluster(n_workers=24,
#                                threads_per_worker=1,
#                                memory_limit='1GB')
#         client = Client(cluster)
#
#
#         logging.info(f"Dask Dashboard is running at {client.dashboard_link}")
#
#
#         input("Press enter to exit...")
#
#     except Exception as e:
#         logging.error("Error starting LocalCluster", exc_info=True)

app.layout = global_layout()

register_callbacks(app)

app.clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute("data-bs-theme", switchOn ? "light" : "dark"); 
       return window.dash_clientside.no_update
    }
    """,
    Output("switch", "id"),
    Input("switch", "value"),
)


if __name__ == '__main__':
    # cluster()
    app.run_server(debug=False, host='0.0.0.0', port=8051)
