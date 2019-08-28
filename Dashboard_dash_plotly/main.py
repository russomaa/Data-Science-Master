# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:48:04 2019

Digitalisation in the European Union

@authors: Guillermo Gliment, Ruben Gimenez, Ana Cristina Ros & Mayra Russo
"""
import pandas as pd
import numpy as np
import json

import dash_table
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_colorscales

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

import folium
import geocoder
import os

import colorlover as cl
from IPython.display import HTML

## DATA ##
data_path = "data/dash_smartcities.xls"
country_codes_path = "data/country_codes.csv"
country_codes = pd.read_csv(country_codes_path, sep = ",", header = 0)

country_names = pd.read_excel(data_path, sheet_name = 0, header = 0)
desi = pd.read_excel(data_path, sheet_name = 1, header = 0)
# rearrange desi
desi = desi.loc[:,["country", 2014, 2015, 2016, 2017, 2018]].reindex()
desi_breakdown = pd.read_excel(data_path, sheet_name = 2, header = 0)
# rearrange desi_breakdown
desi_breakdown = desi_breakdown.loc[:, ["series", "country", 2014, 2015, 2016, 2017, 2018]].reindex()
# gdp
gdp = pd.read_excel(data_path, sheet_name = 3, header = 0)
# population
population = pd.read_excel(data_path, sheet_name = 4, header = 0)
#table summary
sum_table = pd.read_excel(data_path, sheet_name = 5, header = 0)
# key indicators (rank)
rank = pd.read_excel(data_path, sheet_name = 6, header = 0)

# unemployment
unemp = pd.read_excel(data_path, sheet_name = 7, header = 0)
# DEBT
debt = pd.read_excel(data_path, sheet_name = 8, header = 0)
# Internet coverage
intcov = pd.read_excel(data_path, sheet_name = 9, header = 0)
# digital education
digiedu = pd.read_excel(data_path, sheet_name = 10, header = 0)
# Investment
invest = pd.read_excel(data_path, sheet_name = 11, header = 0)
# E-goverment
egov = pd.read_excel(data_path, sheet_name = 12, header = 0)
# E-health
eheal = pd.read_excel(data_path, sheet_name = 13, header = 0)
# ageing
aging = pd.read_excel(data_path, sheet_name = 14, header = 0)

#add country codes to data
dataframes = [country_names, desi, desi_breakdown, gdp, population, rank,debt,invest,egov,eheal,intcov,digiedu,unemp,aging]

for i in range(len(dataframes)):
    dataframes[i] = pd.merge(dataframes[i], country_codes, on = "country")
country_names = dataframes[0]
desi = dataframes[1]
desi_breakdown = dataframes[2]
gdp = dataframes[3]
population = dataframes[4]
rank = dataframes[5]
debt = dataframes[6]
invest = dataframes[7]
egov = dataframes[8]
eheal = dataframes[9]
intcov = dataframes[10]
digiedu = dataframes[11]
unemp = dataframes[12]
ageing = dataframes[13]


# Load leafelt map geometry
# load geometry
geometry_path = "data/world-countries.json"
country_geo = json.load(open(geometry_path, "r"))
# save country codes from our dataset
eu_countrycodes = desi.code.tolist()
# select indices
country_geo_features  = country_geo["features"]
eu_country_geo = []
for i in range(len(country_geo_features)):
    if country_geo_features[i]["id"] in eu_countrycodes:
        aux = dict((k, v) for k, v in country_geo_features[i].items())
        eu_country_geo.append(aux)
country_geo["features"] = eu_country_geo

# config
YEARS = desi.select_dtypes(["float64","int64"]).columns.values
COUNTRIES = desi["country"].unique()[:-1] # avoid EU28
SERIES = desi_breakdown["series"].unique().tolist()

BINS = ["0-2", "2.1-4", "4.1-6", "6.1-8", "8.1-10", "10.1-12", "12.1-14", \
		"14.1-16", "16.1-18", "18.1-20", "20.1-22", "22.1-24",  "24.1-26", \
		"26.1-28", "28.1-30", ">30"]

DEFAULT_COLORSCALE = ["#2a4858", "#265465", "#1e6172", "#106e7c", "#007b84", \
	"#00898a", "#00968e", "#19a390", "#31b08f", "#4abd8c", "#64c988", \
	"#80d482", "#9cdf7c", "#bae976", "#d9f271", "#fafa6e"]
BARPLOT_COLORSCALE = [DEFAULT_COLORSCALE[i*3] for i in range(len(YEARS))]

DEFAULT_OPACITY = 0.8

bluespalette= ["#ADD8E6", "#63D1F4", "#0EBFE9","#C1F0F6", "#0099CC"]

## FUNCTIONS ##

def indicator(color, text, id_value):
    return html.Div(
        [
            html.P(
                text,
                className="twelve columns indicator_text" # QUITAR NO HASE NA
            ),
            html.P(
                id = id_value,
                className="indicator_value"# QUITAR NO HASE NA
            ),
        ],
        className="four columns indicator"#, style = {"background-color": "rgb(255, 153, 51)"},# QUITAR NO HASE NA
    )



### DASHBOARD APP ###

app = dash.Dash()

# Boostrap CSS.
app.css.append_css({"external_url": "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css"}) # noqa: E501

colors = { #colour selected for titles
    "background": "#111111", #"#4E44A6",#"#15078F",
    "text":  "#262626"#"#7FDBFF" #"#01DFFE"
}



## LAYOUT ##

app.layout = html.Div([

    # Title
    html.Div(
        [
            html.H1(
                "Digitalisation in the European Union",
                style = {
                    "text-align": "center",
                    "font-family": "Calibri Light",
                    'color':colors['text']
                }
            ),
        ],
        className = "row",
    ),

    html.Br(),

    # First row
    html.Div([
        # Country dropdown
        html.Div(
            [
                html.P(
                    "Select country and year",
                    style = {
                        "text-align": "center",
                        "font-family": "Calibri",
                        "font-size": "21px",
                        'color':colors['text']
                    },
                ),

                dcc.Dropdown(
                    id="country-dropdown",
                    options=[{"label": i, "value": i } for i in COUNTRIES],
                    value="Spain",
                ),

                html.Div(
                    [
                        dcc.Slider(
                			id = "years-slider",
                			min = min(YEARS),
                			max = max(YEARS),
                			value = min(YEARS),
                			marks = { int(year) : {'label':str(year),'style': {'color': '#262626'}} for year in YEARS},
                		),
                    ],
                    style = {"marginTop": "15",'color': '#82c3f8'},
                ),
            ],
            className = "two columns",
            style = {"marginLeft": "20",'color':colors['text']},
        ),

        # Indices boxes
        html.Div(
            [
                indicator(
                    "#00cc96",
                    "Rank",
                    "left_rank_indicator",
                ),
                indicator(
                    "#119DFF",
                    "GDP per capita growth",
                    "middle_gdp_indicator",
                ),
                indicator(
                    "#EF553B",
                    "Population",
                    "right_pop_indicator",
                ),
            ],
            className = "seven columns",
            style = {
                "text-align":"center",
                "font-family":"Calibri",
                "font-size": 20,
                'color':colors['text'],
            },
        ),

        # Table
        html.Div(
            id = "table",
            className = "two columns"
        ),

    ], className = "row"),


    # Second row
    html.Div([
        html.H3("Digital Economy and Society Index (DESI)"),
    ],className = "row", style = {"text-align": "center", "font-family": "Calibri Light",'color':colors['text']}),
    html.Div(
        [
            # Map
            html.Div(
                [
                    html.Div(id = "folium_map"),
                ],
                className = "six columns",
            ),

            # Bar plots
            html.Div(
                [
                    dcc.Graph(id = "country-barplot")
                ],
                className = "six columns", style = {"height":"100%"}
            ),
        ],
        className = "row",
        style = {"text-align": "center", "font-family": "Calibri Light"},
    ),


    # Third row
    html.Div(
        [
            # Pie charts
            html.Div(
                [
                    dcc.Graph(id = "PieBreakdown"),
                ],
                className = "six columns", style = {"text-align": "center", "font-family": "Calibri Light"},
            ),

            # Macro-economic chart
            html.Div(
                [
                    dcc.Graph(id = "economic-linechart")
                ],
                className = "six columns",
            ),

    ], className = "row"),


    # fourth row
    html.Div(
        [
            #line charts
            html.Div(
                [
                    dcc.Graph(id = "digital-linechart"),
                ],
                className = "six columns", style = {"text-align": "center", "font-family": "Calibri Light"},
            ),

            # Macro-economic chart
            html.Div(
                [
                    dcc.Graph(id =  "social-linechart")
                ],
                className = "six columns",
            ),

    ], className = "row"),

], style = {"background-color": "rgb(244, 245, 247)"})



## CALLBACK FUNCTIONS ##

@app.callback(
		Output("country-barplot", "figure"),
		[Input("country-dropdown", "value"),
        Input("years-slider", "value")])
def update_country_barplot(country, years):
    df = desi.sort_values([years], ascending= False).reset_index(drop=True)
    base_color = np.repeat("rgb(70, 163, 255)", len(df.country))
    # highlight eu28 position
    eu28_index = df[df["country"] == "EU28"].index[0]
    base_color[eu28_index] = "rgb(77, 77, 255)"
    # highlight selected country composition
    selected_country = df[df["country"] == country].index[0]
    base_color[selected_country] = "rgb(255, 136, 77)"
    # graph
    trace0 = go.Bar(
                x = df["country"].values,
                y = df[years].values,
                marker = dict(color = base_color),
                opacity = 0.8,
            )
    data = [trace0]
    layout = go.Layout(
                margin=go.layout.Margin(l=30, r=20, t=5, b=100),
                xaxis = dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=True,
                        linecolor="rgb(204, 204, 204)",
                        linewidth=2,
                        ticks="outside",
                        tickcolor="rgb(204, 204, 204)",
                        tickwidth=2,
                        ticklen=5,
                        tickangle  = -45,
                        tickfont = dict(
                            family = "Calibri",
                            size = 14,
                            color = "rgb(82, 82, 82)"
                                        )
                        ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'

            )
    figures=go.Figure(data, layout)

    return {"data": figures}


@app.callback( # macro-economic line chart
		Output("economic-linechart", "figure"),
		[Input("country-dropdown", "value")])
def update_econ_country_linechart(country):
    gdpl = gdp[gdp["country"] == country] if country else gdp[gdp["country"] == "EU28"]
    debtl = debt[debt["country"] == country] if country else debt[debt["country"] == "EU28"]
    investl = invest[invest["country"] == country] if country else invest[invest["country"] == "EU28"]
    linedata=[gdpl,debtl,investl]
    names=["GDP per Capita Growth","Debt Growth","Investment Growth"]
    line_colors = ["rgb(70, 163, 255)", "rgb(255, 83, 83)", "rgb(60, 255, 123)"]
    marker_color = "rgb(60, 60, 60)"
    data = [go.Scatter(
                x = YEARS,
                y = (linedata[i].iloc[0,1:-1].values)/100,
                name = names[i],
                connectgaps = True,
                hoverinfo = "name+ y"
                ,hovertemplate = "%{y:.2%}",
                marker = dict(color = marker_color, size = 6),
                line = dict(color = line_colors[i], width = 3),
                opacity = DEFAULT_OPACITY,
            )for i in range(len(linedata))]

    layout=go.Layout(
             title=dict(text = "Macro-Economic Indicators", font = dict(family = "Calibri Light", size = 28)),
             showlegend=True,
             legend=dict(orientation="h", x = 0.12),
             font = dict(family="Calibri", size=14),
             margin=go.layout.Margin(l=30, r=20, t=60, b=30),
             xaxis = dict(
                     tickmode = "auto",
                     nticks = 5,
                     showline=True,
                     showgrid=True,
                     showticklabels=True,
                     linecolor="rgb(204, 204, 204)",
                     linewidth=2,
                     ticks="outside",
                     tickcolor="rgb(204, 204, 204)",
                     tickwidth=2,
                     ticklen=5,
                     tickfont = dict(
                         family = "Calibri",
                         size = 14,
                         color = "rgb(82, 82, 82)"
                     )
            ),
            yaxis = dict(
                    showgrid = False,
                    zeroline = True,
                    zerolinecolor = "rgb(204, 204, 204)",
                    zerolinewidth = 1,
                    showline = False,
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
    )
    figures=go.Figure(data,layout)
    return {"data": figures}

@app.callback( # digital line chart
	Output("digital-linechart", "figure"),
	[Input("country-dropdown", "value")])
def update_digit_country_linechart(country):
    egovl = egov[egov["country"] == country] if country else egov[egov["country"] == "EU28"]
    eheall = eheal[eheal["country"] == country] if country else eheal[eheal["country"] == "EU28"]
    intcovl = intcov[intcov["country"] == country] if country else intcov[intcov["country"] == "EU28"]
    linedata=[egovl,eheall,intcovl]
    names=["e-Government Index","e-Health Index","Internet Coverage Index"]
    line_colors = ["rgb(70, 163, 255)", "rgb(255, 83, 83)", "rgb(60, 255, 123)"]
    marker_color = "rgb(60, 60, 60)"
    data = [go.Scatter(
                x = YEARS,
                y = (linedata[i].iloc[0,1:-1].values)/100,
                name = names[i],
                connectgaps = True,
                hoverinfo = "name+ y"
                ,hovertemplate = "%{y:.2r}"
                ,marker = dict(color = marker_color, size = 6),
                line = dict(color = line_colors[i], width = 3),
                opacity = DEFAULT_OPACITY,
            )for i in range(len(linedata))]

    layout=go.Layout(
             title=dict(text = "Digital Indicators (Growth %)", font = dict(family = "Calibri Light", size = 28)),
             showlegend=True,
             legend=dict(orientation="h", x = 0.12),
             font = dict(family="Calibri", size=14),
             margin=go.layout.Margin(l=30, r=20, t=60, b=30),
             xaxis = dict(
                     tickmode = "auto",
                     nticks = 5,
                     showline=True,
                     showgrid=True,
                     showticklabels=True,
                     linecolor="rgb(204, 204, 204)",
                     linewidth=2,
                     ticks="outside",
                     tickcolor="rgb(204, 204, 204)",
                     tickwidth=2,
                     ticklen=5,
                     tickfont = dict(
                         family = "Calibri",
                         size = 14,
                         color = "rgb(82, 82, 82)"
                     )
            ),
            yaxis = dict(
                    showgrid = False,
                    zeroline = True,
                    zerolinecolor = "rgb(204, 204, 204)",
                    zerolinewidth = 1,
                    showline = False,
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
    )
    figures=go.Figure(data,layout)
    return {"data": figures}

@app.callback( # social line chart
		Output("social-linechart", "figure"),
		[Input("country-dropdown", "value")])
def update_soc_country_linechart(country):
    digiedul = digiedu[digiedu["country"] == country] if country else digiedu[digiedu["country"] == "EU28"]
    unempl = unemp[unemp["country"] == country] if country else unemp[unemp["country"] == "EU28"]
    ageingl = ageing[ageing["country"] == country] if country else ageing[ageing["country"] == "EU28"]
    linedata=[digiedul ,unempl ,ageingl]
    names=["Digital Education Rate","Unemployment Rate","Ageing Rate"]
    line_colors = ["rgb(70, 163, 255)", "rgb(255, 83, 83)", "rgb(60, 255, 123)"]
    marker_color = "rgb(60, 60, 60)"
    data = [go.Scatter(
                x = YEARS,
                y = (linedata[i].iloc[0,1:-1].values)/100,
                name = names[i],
                connectgaps = True,
                hoverinfo = "name+ y"
                ,hovertemplate = "%{y:.2r}"
                ,marker = dict(color = marker_color, size = 6),
                line = dict(color = line_colors[i], width = 3),
                opacity = DEFAULT_OPACITY,
            )for i in range(len(linedata))]

    layout=go.Layout(
             title=dict(text = "Social Indicators (Growth %)", font = dict(family = "Calibri Light", size = 28)),
             showlegend=True,
             legend=dict(orientation="h", x = 0.12),
             font = dict(family="Calibri", size=14),
             margin=go.layout.Margin(l=30, r=20, t=60, b=30),
             xaxis = dict(
                     tickmode = "auto",
                     nticks = 5,
                     showline=True,
                     showgrid=True,
                     showticklabels=True,
                     linecolor="rgb(204, 204, 204)",
                     linewidth=2,
                     ticks="outside",
                     tickcolor="rgb(204, 204, 204)",
                     tickwidth=2,
                     ticklen=5,
                     tickfont = dict(
                         family = "Calibri",
                         size = 14,
                         color = "rgb(82, 82, 82)"
                     )
            ),
            yaxis = dict(
                    showgrid = False,
                    zeroline = True,
                    zerolinecolor = "rgb(204, 204, 204)",
                    zerolinewidth = 1,
                    showline = False,
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
    )
    figures=go.Figure(data,layout)
    return {"data": figures}

@app.callback( # table
    dash.dependencies.Output(component_id = "table",component_property = "children"),
    [dash.dependencies.Input(component_id = "country-dropdown", component_property = "value")])
def table_selector(country):
    my_table = sum_table.loc[:,["Country", country]]
    new_header = my_table.iloc[0]
    my_table = my_table[1:]
    my_table.columns = new_header
    return html.Div([
        dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in my_table.columns],
            data=my_table.to_dict("rows"),
            style_as_list_view=False,
            style_cell={
                "textAlign": "left",
                "font-family":"Calibri",
                },
        )

    ],style={"maxHeight": "350px",
            "padding": "5",
            "font-family":"Calibri",
            "marginTop": "5",
            "font-size":"18",
            "backgroundColor":"white",
            "border": "1px solid #C8D4E3",
            "borderRadius": "3px",
            "width":"300px",
            "align":"center",
            "margin":"18"})

@app.callback( # THE map
    Output(component_id = "folium_map",component_property = "children"),
    [Input(component_id = "country-dropdown", component_property = "value"),
    Input(component_id = "years-slider", component_property = "value")])
def map_creator(country, year):
    g = geocoder.osm(country)
    lon = g.osm["x"]
    lat = g.osm["y"]
    coords= [lat, lon]
    m = folium.Map(location=[lat, lon], zoom_start=4, tiles="OpenStreetMap")
    tooltip = "Click for index!"
    rowindex = desi[desi["country"] == country].index.values
    popup = "DESI: " + str(round(desi.loc[rowindex,int( year)].values[0], 2))
    folium.Marker(coords, popup="<i>" + popup + "</i>", tooltip=tooltip).add_to(m)
    folium.Choropleth(
        geo_data=country_geo,
        name="choropleth",
        data=desi,
        columns=["code", int(year)],
        key_on="feature.id",
        fill_color= "YlGnBu" ,
        fill_opacity=0.8,
        line_opacity=0.1,
        legend_name="Digital Economy and Society Index (pts)"
    ).add_to(m)

    m.save("index.html")
    return html.Div([
            html.Iframe(srcDoc = open("index.html", "r").read(),style={"width":"100%", "height":400})
    ])

@app.callback(
    [Output("left_rank_indicator", "children"),
     Output("middle_gdp_indicator", "children"),
     Output("right_pop_indicator", "children")],
    [Input("country-dropdown", "value"),
     Input("years-slider", "value")])
def update_indices(country, year):
    selected_rank = rank[rank["country"] == country][year]
    # edit
    selected_rank = "#" + str(selected_rank.values.flatten()[0])
    selected_gdp = gdp[gdp["country"] == country][str(year)]
    # edit
    selected_gdp = str(selected_gdp.values.flatten()[0]) + "%"
    selected_pop = population[population["country"] == country][str(year)]
    # edit
    selected_pop = str(selected_pop.values.flatten()[0]) + "M"

    return (selected_rank, selected_gdp, selected_pop)

#### PIE CHARTS ####
@app.callback(
		Output("PieBreakdown", "figure"),
		[Input("country-dropdown", "value"),
       Input("years-slider", "value")])
def update_country_pieplot(country,years):
    df = desi_breakdown[desi_breakdown["country"] == country] if country else desi_breakdown
    if df["country"].values[0] == "Czech Republic":
        df.loc[:, "country"] = "Czechia"
    elif df["country"].values[0] == "United Kingdom":
        df.loc[:, "country"] = "UK"
    df_fijo=desi_breakdown[desi_breakdown["country"] == "EU28"]
    traces = tools.make_subplots(rows = 1, cols = 2, subplot_titles=("Plot 1", "Plot 2"))
    trace1 = go.Pie( #country pieplots
                labels=df["series"].values,
                values=df.loc[:,years].values.flatten(),
                name=str(years)
                ,showlegend=True
                ,domain= {"x": [0.2, 0.45], "y": [0.2, 0.8]}
                ,hole=0.5
                ,textposition="inside"
                ,marker=dict(colors=bluespalette)
                ,hoverinfo="label+value"
                ,textinfo="percent"
                ,textfont=dict(size=12)
                ,outsidetextfont=dict(size=20)
            )
    trace2 = go.Pie( # eu28 pie chart
                labels=df_fijo["series"].values,
                values=df_fijo.loc[:,years].values.flatten(),
                name=years
                ,showlegend=True
                ,domain=  {"x": [0.55, 0.8], "y": [0.2, 0.8]}
                ,hole=0.5
                ,textposition="inside"
                ,marker=dict(colors=bluespalette)
                ,hoverinfo="label+value"
                ,textinfo="percent"
                ,textfont=dict(size=12)
                ,outsidetextfont=dict(size=20)
            )
    layout=go.Layout(
             title=dict(text = "DESI Breakdown Decomposition"),
             showlegend=True,
             legend=dict(orientation="h", x = 0.12),
             font = dict(family="Calibri", size=14),
             paper_bgcolor='rgba(0,0,0,0)',
             plot_bgcolor='rgba(0,0,0,0)'
             ,annotations=[
                dict(
                    text = "DESI Breakdown Decomposition",
                    font = dict(family = "Calibri Light", size = 28),
                    align = "center",
                    showarrow = False,
                    y = 1
                ),
                dict(font = dict(size= 14),
                        showarrow = False,
                        text = str(df["country"].values[0]),
                        x = 0.3,
                        y = 0.5
                    ),
                    # subtitle location:
                dict(font = dict(size= 14),
                        showarrow = False,
                        text = "EU-28",
                        x = 0.698,
                        y = 0.5
                    )
                ]
             ,margin=go.layout.Margin(l=0, r=0, t=0, b=0)
                 )
    traces = go.Data([trace1,trace2])
    figures=go.Figure(traces,layout)
    return {"data": figures}

## RUN SERVER ##
if __name__ == "__main__":
    app.run_server(debug=True)
