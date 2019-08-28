import dash_table
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_colorscales
import pandas as pd
import numpy as np
import folium
import geocoder
import os
##https://georgetsilva.github.io/posts/mapping-points-with-folium/
#prepare data
# data = pd.read_csv("data/coordenadas.csv")
# #data.head()
# #list of coordinates
# locations=data[['latitude','longitude']]
# locationslist=locations.values.tolist()
# len(locationslist)
# locationslist[7]
# #map
# endgame_map = folium.Map(location=[14.43369519,121.01080974],tiles='Stamen Toner',zoom_start=2)
# for point in range(0, len(locationslist)):
#     folium.Marker(locationslist[point], popup=data['user'][point],icon=folium.Icon(icon='dot',color='green')).add_to(endgame_map)
# #save map in html
# avenge_map.save('endgame_map.html')



### DASHBOARD APP ###

app = dash.Dash()

# Boostrap CSS.
app.css.append_css({"external_url": "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css"}) # noqa: E501

app.layout = html.Div([
      html.H1("#AvengersEndGame Opening Week Tweets",
               style = {
                    "text-align": "center",
                    "font-family": "Sans‑serif"
                }),
      html.Iframe(srcDoc = open('endgame_map.html', 'r').read(),style={"width":"100%", "height":500})
], className = "ten columns offset-by-one", style={"align-self":"center"})

## RUN SERVER ##
if __name__ == "__main__":
    app.run_server(debug=True)
