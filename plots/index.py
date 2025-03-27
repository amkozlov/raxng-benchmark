# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import pathlib

from app import app
from layouts import layout
import init_callbacks
import callbacks

app.layout = layout

if __name__ == "__main__":
    app.run_server(host= '0.0.0.0', debug=True)
#    app.run_server(debug=False)

