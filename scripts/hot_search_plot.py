import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import os
import plotly.graph_objects as go
import datetime
from datetime import date, timedelta
import numpy as np


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# plot heatmap
map_df = pd.read_csv("./hot_search_data/hot_topic_heatmap.csv", sep=',')
y = list(map_df.columns)[2:]

begin = datetime.date(2020,1,1)
end = datetime.date(2020, 3, 6)
delta = end - begin 
x = []
for i in range(delta.days + 1):
    x.append(begin + timedelta(days=i))

z = []
for i in range(len(y)):
    z_part = map_df[y[i]].values.tolist()
    z.append(z_part)
   
date = map_df['start_mnth_day']
start_default = date.iloc[0]
end_default = date.iloc[-1]

# plot top20 hot topic
start_date = date.copy()
start_date = start_date.append(pd.Series({'start_mnth_day': 'Total'}) ,ignore_index=True)

end_date = date.copy()
end_date = end_date.append(pd.Series({'start_mnth_day': 'None'}) ,ignore_index=True)

top_20_df = pd.read_csv("./hot_search_data/top20_hot_topic.csv", sep=',')
x_count = top_20_df['count'][::-1]
inside_text = top_20_df['inside_text'][::-1]
rank = top_20_df['rank'][::-1]

color_list =[]
percent = 30
for i in range(len(x_count)):
    color_list.append('hsl(33, 100%, {}%)'.format(percent))
    percent += 3
    
df = pd.read_csv("./hot_search_data/hot_with_live_time.csv", sep=',')

# plot words frequency pie chart
count_df = pd.read_csv("./hot_search_data/translate_words_frequency_count.csv", sep=',')
labels1 = count_df['flatten_words'].values.tolist()
values1 = count_df['count'].values.tolist()

alivetime_df = pd.read_csv("./hot_search_data/translate_words_frequency_alivetime.csv", sep=',')   
labels2 = alivetime_df['flatten_words'].values.tolist()
values2 = alivetime_df['alive_time'].values.tolist()


app.layout = html.Div(children=[
    html.H1(style={'font-size':'35px','margin-left': 'auto', 
                   'margin-right': 'auto', 'margin-top': 'auto', 
                   'textAlign': 'center'}, 
            children='Weibo Hot Search Analysis'),
    
    html.Div([
        html.H1(style={'font-size': '25px', 'color':'#97A09B', 
                       'margin-left': 'auto', 'margin-right': 'auto', 
                       'margin-top': 'auto'},
                children='Number of Hot Search Alive Time and View Per Day'),
        html.Div([
            html.Div([html.H4('Select Start Date'),
                      dcc.Dropdown(id='startdate_dropdown',
                                   options=[{'label': i, 'value': i} for i in date],
                                   value = start_default),],
                     style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([html.H4('Select End Date'),
                      dcc.Dropdown(id='enddate_dropdown',
                                   options=[{'label': i, 'value': i} for i in date],
                                   value = end_default),],
                     style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            
            dcc.Graph(id='heatmap_down', 
                     figure = {'data': [go.Heatmap(x=x,
                                                   y=['Visitors', 'Alive Time'],
                                                   z=z,
                                                   name = 'first legend group',
                                                   colorscale='Blues')],
                               'layout': {'title': {'text': "Hot Search Heatmap Between " + 
                                                    start_default + " And " + end_default,
                                                    'xanchor': 'center',
                                                    'yanchor': 'top',},  
                                        'font': dict(family="Arial", color="#262626"),
                                        'xaxis_nticks' : 36
                                        }
                })],)],
             
    style={'width': '60%', 'padding': '0 20','margin-left': '250px', 
           'margin-right': '250px', 'margin-top': '50px'}),
    
    html.Div([
        html.H1(style={'font-size': '25px', 'color':'#97A09B', 
                       'margin-left': 'auto', 'margin-right': 'auto', 
                       'margin-top': 'auto'},
                children='Top 20 Hot Search'),
        html.Div([
            html.Div([html.H4('Select Start Date'),
                      dcc.Dropdown(id='top20_startdate',
                                   options=[{'label': i, 'value': i} for i in start_date],
                                   value = 'Total'),],
                     style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([html.H4('Select End Date'),
                      dcc.Dropdown(id='top20_enddate',
                                   options=[{'label': i, 'value': i} for i in end_date],
                                   value = 'None'),],
                     style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
            
            dcc.Graph(id='top20_hot_topic',
                      className='my_graph',
                      figure={'data': [{'x': x_count,
                                        'y': rank,
                                        'type': 'bar',
                                        'orientation': 'h',
                                        'marker': {'color': color_list,},
                                        'text': inside_text,
                                        'textposition': 'inside',
                                        'insidetextanchor': 'middle',
                                        'textfont': {'family':'Arial', 
                                                     'size': 18,
                                                     'color': '#262626'},
                                    }],
                              'layout': {'title': {'text': "Total Top 20 Hot Search",
                                                   'y':0.92,'x':0.5,
                                                   'xanchor': 'center',
                                                   'yanchor': 'top',},
                                         'yaxis': {'hoverformat': '.0f'},
                                         
                                         'font': dict(family="Arial", color="#262626"),
                                         'plot_bgcolor': '#ffffff',
                                         'xaxis': dict(showticklabels=False, gridcolor='#ffffff',zeroline=False,),
                                         'yaxis': dict(tickfont=dict(color='#262626',
                                                                     size = 12,),),
                                     }
                          },
                  config={'displayModeBar': False},
                  ),],),],
    style={'width': '60%', 'padding': '0 20','margin-left': '250px', 
           'margin-right': '250px', 'margin-top': '50px'}),
      
    html.Div([
        html.H1(style={'font-size': '25px', 'color':'#97A09B', 
                       'margin-left': 'auto', 'margin-right': 'auto', 
                       'margin-top': 'auto'},
                children='Top 20 Hot Search Words Frequency'), 
        html.Div([
            dcc.RadioItems(
                id='language',
                options=[{'label': i, 'value': i} for i in ['Chinese', 'English']],
                value='Chinese',
                labelStyle={'display': 'inline-block'}
            ),
           
            html.Div([
                dcc.Graph(id='words_frequency_view', 
                          figure = {'data': [go.Pie(labels=labels1,
                                                    values=values1,
                                                    textinfo='label+percent',)], 
                                    'layout': {'title':
                                        {'text': "Top 20 Words Frequency(Total Views)",
                                         'y':0.92,'x':0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',},
                                         'font': dict(family="Arial",color="#262626"),
                                         'autosize': True}              
                            },),
                ]),
            
            html.Div([
                dcc.Graph(id='words_frequency_alivetime', 
                          figure = {'data': [go.Pie(labels=labels2,
                                                    values=values2,
                                                    textinfo='label+percent',)],
                                    'layout': {'title':
                                        {'text': "Top 20 Words Frequency(Total Alive Time)",
                                         'y':0.92,'x':0.5,
                                         'xanchor': 'center',
                                         'yanchor': 'top',},
                                        'font': dict(family="Arial",color="#262626"),
                                        'autosize': True}     
                            },),
                ],),   
            ],),],      
    style={'width': '60%', 'padding': '0 20','margin-left': '250px', 
           'margin-right': '250px', 'margin-top': '50px'}),         
])


@app.callback(
    dash.dependencies.Output('heatmap_down', 'figure'),
    [dash.dependencies.Input('startdate_dropdown', 'value'),
     dash.dependencies.Input('enddate_dropdown', 'value')])
def update_graph(startdate, enddate):  
    if enddate > startdate:    
        df = map_df[map_df.start_mnth_day.between(startdate, enddate)]
        x_new = df['start_mnth_day']
        y_new = list(df.columns)[2:]
        z_new = []
        
        for i in range(len(y_new)):
            z_part = df[y_new[i]].values.tolist()
            z_new.append(z_part)
        
        return {
            'data': [go.Heatmap(x=x_new,
                                y=['Visitors', 'Alive Time'],
                                z=z_new,
                                xgap = 2,
                                ygap = 2,
                                colorscale='Blues')],     
            'layout': {'title': {'text': "Hot Topic Heatmap between " + startdate + " And " + enddate,
                                 'xanchor': 'center',
                                 'yanchor': 'top',
                                 },  
                       'font': dict(family="Arial",color="#262626"),
                       'xaxis_nticks' : 36
                       }
        }
    
    
@app.callback(
    dash.dependencies.Output('top20_hot_topic', 'figure'),
    [dash.dependencies.Input('top20_startdate', 'value'),
     dash.dependencies.Input('top20_enddate', 'value')])
def update_bar(startdate, enddate):  
    
    if enddate == 'None':
        if startdate == 'Total':
            return{
                'data': [{'x': x_count,
                          'y': rank,
                          'type': 'bar',
                          'orientation': 'h',
                          'marker': {'color': color_list,},
                          'text': inside_text,
                          'textposition': 'inside',
                          'insidetextanchor': 'middle',
                          'textfont': {'family':'Arial', 
                                       'size': 18,
                                       'color': '#262626'},
                          }],
                'layout': {'title': 
                    {'text': "Total Top 20 Hot Search",
                     'xanchor': 'center',
                     'yanchor': 'top',},
                     'height': 600,
                     'yaxis': {'hoverformat': '.0f'},
                     'margin': {'l': 80, 'r': 35, 't': 50, 'b': 80},
                     'plot_bgcolor': '#ffffff',
                     'xaxis': dict(showticklabels=False, 
                                   gridcolor='#ffffff',
                                   zeroline=False,),
                     'yaxis': dict(tickfont=dict(color='#262626',
                                                 size = 12,),),
                     } 
                }
            
        else:
            df_new = df[df.start_mnth_day == startdate]
            df_new = df_new.sort_values(by=['count'], ascending=False)
            df_new = df_new.iloc[:20]
            
            length = df_new.count()[0]
            rank_list = ["NO." + str(num) for num in [i for i in range(1, length+1)]]
            df_new['rank'] = rank_list

            inside_text_new = []
            for i in range(len(rank_list)): 
                inside_text_new.append("{}   Search count:{}".format(df_new['content'].iloc[i], 
                                                                 df_new['count'].iloc[i]))
            df_new['inside_text'] = inside_text_new

            x_count_new = df_new['count'][::-1]
            inside_text_new = df_new['inside_text'][::-1]
            rank_new = df_new['rank'][::-1]
    
            return{
                'data': [{'x': x_count_new,
                          'y': rank_new,
                          'type': 'bar',
                          'orientation': 'h',
                          'marker': {'color': color_list,},
                          'text': inside_text_new,
                          'textposition': 'inside',
                          'insidetextanchor': 'middle',
                          'textfont': {'family':'Arial', 
                                       'size': 18,
                                       'color': '#262626'},
                          }],
                
                'layout': {'title': 
                    {'text': "Top 20 Hot Search on " + startdate,
                     'xanchor': 'center',
                     'yanchor': 'top',},
                     'height': 600,
                     'yaxis': {'hoverformat': '.0f'},
                     'margin': {'l': 80, 'r': 35, 't': 50, 'b': 80},
                     'plot_bgcolor': '#ffffff',
                     'xaxis': dict(showticklabels=False, 
                                   gridcolor='#ffffff',
                                   zeroline=False,),
                     'yaxis': dict(tickfont=dict(color='#262626',
                                                 size = 12,),),
                     } 
                }
            
    elif enddate > startdate:
        df_new = df[df.start_mnth_day.between(startdate, enddate)]
        df_new = df_new.sort_values(by=['count'], ascending=False)
        df_new = df_new.iloc[:20]
        
        length = df_new.count()[0]
        rank_list = ["NO." + str(num) for num in [i for i in range(1, length+1)]]
        df_new['rank'] = rank_list

        inside_text_new = []
        for i in range(len(rank_list)): 
            inside_text_new.append("{}   Search count:{}".format(df_new['content'].iloc[i], 
                                                                df_new['count'].iloc[i]))
        df_new['inside_text'] = inside_text_new

        x_count_new = df_new['count'][::-1]
        inside_text_new = df_new['inside_text'][::-1]
        rank_new = df_new['rank'][::-1]

        return{
            'data': [{'x': x_count_new,
                        'y': rank_new,
                        'type': 'bar',
                        'orientation': 'h',
                        'marker': {'color': color_list,},
                        'text': inside_text_new,
                        'textposition': 'inside',
                        'insidetextanchor': 'middle',
                        'textfont': {'family':'Arial', 
                                    'size': 18,
                                    'color': '#262626'},
                        }],
            'layout': {'title': 
                {'text': "Top 20 Hot Search between " + startdate + " and " + enddate,
                    'xanchor': 'center',
                    'yanchor': 'top',},
                    'height': 600,
                    'yaxis': {'hoverformat': '.0f'},
                    'margin': {'l': 80, 'r': 35, 't': 50, 'b': 80},
                    'plot_bgcolor': '#ffffff',
                    'xaxis': dict(showticklabels=False, 
                                gridcolor='#ffffff',
                                zeroline=False,),
                    'yaxis': dict(tickfont=dict(color='#262626',
                                                size = 12,),),
                    } 
            }
       
    
@app.callback(
    dash.dependencies.Output('words_frequency_view', 'figure'),
    [dash.dependencies.Input('language', 'value'),])
def update_graph(language):   
    labels1_new = count_df['flatten_words_en'].values.tolist()
    values1_new = count_df['count'].values.tolist()
    
    if language == 'English':
        return {
            'data': [go.Pie(labels=labels1_new,
                            values=values1_new,
                            textinfo='label+percent',)],
            'layout': {'title':{'text': "Top 20 Words Frequency(Total Views)",
                                'y':0.92,'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',},
                                'font': dict(family="Arial",color="#262626"),}     
        }
    if language == 'Chinese':
        return {
            'data': [go.Pie(labels=labels1,
                            values=values1,
                            textinfo='label+percent',)],
            'layout': {'title':{'text': "Top 20 Words Frequency(Total Views)",
                                'y':0.92,'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',},
                                'font': dict(family="Arial",color="#262626"),}                                 
        }
               
@app.callback(
    dash.dependencies.Output('words_frequency_alivetime', 'figure'),
    [dash.dependencies.Input('language', 'value'),])
def update_graph(language):      
    labels2_new = alivetime_df['flatten_words_en'].values.tolist()
    values2_new = alivetime_df['alive_time'].values.tolist()
     
    if language == 'English':
        return {
            'data': [go.Pie(labels=labels2_new,
                            values=values2_new,
                            textinfo='label+percent',)],
            'layout': {'title':{'text': "Top 20 Words Frequency(Total Alive Time)",
                                'y':0.92,'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',},
                                'font': dict(family="Arial",color="#262626"),}                               
        }   
        
    if language == 'Chinese':
        return {
            'data': [go.Pie(labels=labels2,
                            values=values2,
                            textinfo='label+percent',)],
            'layout': {'title':{'text': "Top 20 Words Frequency(Total Alive Time)",
                                'y':0.92,'x':0.5,
                                'xanchor': 'center',
                                'yanchor': 'top',},
                                'font': dict(family="Arial",color="#262626"),}                               
        }


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
