#!/bin/sh
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_auth
from flask import Flask
import pandas as pd
import datetime
import plotly
import plotly.graph_objs as go

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True
un_pw = [['user_1', 'password_1'], ['user_2', 'password_2'], ['user_3', 'password_3']]
auth = dash_auth.BasicAuth(app, un_pw)
initial_country = 'Ireland'
data  = pd.read_csv('../../../data/graph_data.csv')
intro = '''Interactive website analysing Covid-19 data. The data source is https://www.worldometers.info/coronavirus/''' 
app.layout = html.Div(children=[
html.H1(children='Covid 19 Data Analytics'),
dcc.Markdown(children=intro),
dcc.Dropdown(id='country', options=[{'label': i, 'value': i} for i in list(set(data['country']))], value='Ireland'),
html.H4(id='live-update-text'),
dcc.Graph(id='main_graph',style={'width':1000})
])

@app.callback(Output('live-update-text', 'children'), [Input('country', 'value')])
def update_fund_value(value):
    data  = pd.read_csv('../../../data/graph_data.csv')
    data = data[data['country']==value]
    a=data['date'].iloc[-1]
    b=data['country'].iloc[-1]
    c=data['total_cases'].iloc[-1]
    d=round(data['case_growth_rate'].iloc[-1],2)
    e=data['total_deaths'].iloc[-1]
    summary_dict={'Date':[a], 'Country':[b], 'Total Cases':[c], 'Case growth rate':[f'{d}%'], 'Total Deaths':[e]}
    summary_df = pd.DataFrame.from_dict(summary_dict)
    return generate_table(summary_df)

@app.callback(Output('main_graph', 'figure'), [Input('country', 'value')])
def update_layout(value):
    graph_df = pd.read_csv('../../../data/graph_data.csv')
    graph_df = graph_df[graph_df['country']==value]
    # Create the graph with subplots
    fig = plotly.tools.make_subplots(specs=[[{'secondary_y': True}]])
    fig['layout']['title'] = f'Cumulative total of Covid-19 cases in {value} along with daily rates of increase in new cases'
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 60, 't': 60
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    
    fig.add_trace(
    go.Scatter(
        x = graph_df['date'][-15:],
        y = graph_df['total_cases'][-15:],
        name = 'total cases'
        ))

    fig.add_trace(
    go.Bar(
        x = graph_df['date'][-15:],
        y = graph_df['case_growth_rate'][-15:],
        name = 'growth rate'),
        secondary_y=True
        )

    fig.show()    
    return fig
    


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port='80')