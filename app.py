from dash import Dash
from flask_caching import Cache
import dash_bootstrap_components as dbc


# cache_config = {
#     "CACHE_TYPE": "FileSystemCache",
#     "CACHE_DIR": "cache-directory",
#     "CACHE_DEFAULT_TIMEOUT": 20
# }

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
# cache = Cache(app.server, config=cache_config)
app.config.suppress_callback_exceptions = True

cache = Cache(app.server, config={
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': 'my-new-cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 60*60*24*100,  # 100 days
    'CACHE_THRESHOLD': 2**30  # 2^30 files
})
