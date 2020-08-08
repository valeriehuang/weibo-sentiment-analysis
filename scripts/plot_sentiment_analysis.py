import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, log_loss, hamming_loss, \
    precision_recall_curve, roc_curve, average_precision_score, auc
from sklearn.preprocessing import label_binarize
import dash_html_components as html
import base64
import io


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data = pd.read_csv('data/topic_result/topic_all.csv',index_col=0)
data['date'] = data['time'].apply(lambda x: x[:10])
data = data[data['date'] < '2020-04-01']
data = data[data['date'] >= '2020-01-09']
data = data.sort_values(by=['date'], ascending=False)
data['predict_value'] = data.apply(lambda x: -1 if x['predict'] == '__label__negative' else (1 if x['predict'] == '__label__positive' else 0), axis=1)
data['max_interact'] = data.apply(lambda x: max(x['forward'], x['comment'], x['like']), axis=1)
df = data.reset_index()
date = df['date']
forward = df['forward']
comment = df['comment']
like = df['like']

date = date.sort_values(ascending=True).unique()

''' Trend '''
#trendency score
maxInteract = df[df['max_interact'] > 50]

#trendency proportion
def get_proportion_df(df):
    negative_df = df[df['predict'] == '__label__negative']
    positive_df = df[df['predict'] == '__label__positive']
    neutral_df = df[df['predict'] == '__label__neutral']
    all_count = df.groupby('date')['id'].count().reset_index()
    negative_count = negative_df.groupby('date')['id'].count().reset_index()
    positive_count = positive_df.groupby('date')['id'].count().reset_index()
    neutral_count = neutral_df.groupby('date')['id'].count().reset_index()
    count_df = pd.merge(all_count, negative_count, on='date', how='left')
    count_df = pd.merge(count_df, positive_count, on='date', how='left')
    count_df = pd.merge(count_df, neutral_count, on='date', how='left')
    count_df.columns = ['date', 'count', 'negative_count', 'positive_count', 'neural_count']
    count_df['negative_pro'] = count_df.apply(lambda x: x['negative_count']/x['count'], axis=1)
    count_df['positive_pro'] = count_df.apply(lambda x: x['positive_count'] / x['count'], axis=1)
    count_df['neural_pro'] = count_df.apply(lambda x: x['neural_count'] / x['count'], axis=1)

    return count_df

all_proportion_df = get_proportion_df(df)
maxInteract_proportion_df = get_proportion_df(maxInteract)


'''Model Evaluation'''
test_data = df[df['date'] >= '2020-01-27']
test_data = test_data[test_data['date'] <= '2020-03-14']
date_test = test_data['date'].unique()

#average_value
test_data['label_value'] = test_data.apply(lambda x: -1 if x['label'] == '__label__negative' else (1 if x['label'] == '__label__positive' else 0), axis=1)
avg_data = test_data.groupby('date').mean()
avg_score_test = avg_data['predict_value']
avg_label_test = avg_data['label_value']

#confidence
confidence_df = test_data[test_data['label_value']==test_data['predict_value']]
confidence = confidence_df.groupby('date')['predict_score'].mean()

#F-score/Hamming_LOSS
groupby_date = test_data.groupby('date')['predict_value','label_value']
f1_score_value = groupby_date.apply(lambda x: f1_score(x['label_value'], x['predict_value'], average='micro'))
hamming_loss_value = groupby_date.apply(lambda x: hamming_loss(x['label_value'], x['predict_value']))

#PR_curve
y_score = test_data[['__label__negative', '__label__neutral', '__label__positive']].to_numpy()
y_test = test_data['predict_value']
y_test = label_binarize(y_test, classes=[-1, 0, 1])

n_classes = y_test.shape[1]
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,average="micro")

#ROC_curve
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Visualization
app.layout = html.Div(children=[

    html.H1(style={'font-size':'40px','margin-left': '250px', 'margin-right': 'auto', 'margin-top': 'auto'}, children='Weibo Sentiment Analysis'),

    html.Div([
         html.H1(style={'font-size': '30px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='1. Model Evaluation'),
    ], style={'width': '60%', 'margin-left': '250px', 'margin-right': 'auto', 'padding': '0 20'}),

    html.Div([
         html.H1(style={'font-size': '17px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='Trend value: average value of total predict/label value (negative as -1, neutral as 0, positive as 1)'),
         html.H1(style={'font-size': '17px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='F1-score: 2*precision*recall / (precision+recall)'),
         html.H1(style={'font-size': '17px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='Hamming loss: the fraction of the wrong laabels to the total number of labels'),
         html.H1(style={'font-size': '17px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='Confidence: predict probability of test data that predict accurate'),
    ], style={'width': '60%', 'margin-left': '250px', 'margin-right': 'auto', 'padding': '0 20'}),

    html.Div([
         dcc.Graph(id='ME_sentiment_score',
            figure={
                'data': [
                    {'x': date_test, 'y': avg_score_test, 'type': 'line', 'name': 'predict value'},
                    {'x': date_test, 'y': avg_label_test, 'type': 'line', 'name': 'label value'},

                ],
                'layout': {
                    'title': 'Predict and label trend',
                    'xaxis': {"title": 'date'},
                    'yaxis': {"title": 'trend value'},
                    'height': 600,
                    'margin': {'l': 40, 'b': 100, 'r': 10, 't': 100}
                }
            }),
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
         dcc.Graph(id='ME_Fscore',
                figure={
                'data': [
                    {'x': date_test, 'y': f1_score_value, 'type': 'line', 'name': 'predict_score'},
                ],
                'layout': {
                    'title': 'F1-Score',
                    'xaxis': {"title": 'date'},
                    'yaxis': {"title": 'Fscore'},
                    'height': 200,
                    'margin': {'l': 40, 'b': 20, 'r': 10, 't': 50}
                }
            }),
         dcc.Graph(id='ME_hamming_loss',
                figure={
                'data': [
                    {'x': date_test, 'y': hamming_loss_value, 'type': 'line', 'name': 'predict_score'},
                ],
                'layout': {
                    'title': 'Hamming Loss',
                    'xaxis': {"title": 'date'},
                    'yaxis': {"title": 'Hamming Loss'},
                    'height': 200,
                    'margin': {'l': 40, 'b': 20, 'r': 10, 't': 50}
                }
            }),
         dcc.Graph(id='ME_confidence',
                   figure={
                       'data': [
                           {'x': date_test, 'y': confidence, 'type': 'line', 'name': 'predict_score'},

                       ],
                       'layout': {
                           'title': 'Confidence',
                           'xaxis': {"title": 'date'},
                           'yaxis': {"title": 'Confidence'},
                           'height': 200,
                           'margin': {'l': 40, 'b': 20, 'r': 10, 't': 50}
                       }
             }),
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        html.H1(style={'font-size': '20px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='ROC and PR Curve of model'),
        html.H1(style={'font-size': '17px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='True Positvie Rate = TP/TP+FN'),
        html.H1(style={'font-size': '17px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='False Positvie Rate = FP/FP+TN'),
        html.H1(style={'font-size': '17px', 'color':'#97A09B', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 'auto'},
                children='ROC/PR curve is presented by micro average of 3 classes or each class respectively'),
    ], style={'width': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '0 20'}),

    html.Div([

        html.Div([html.H4('Select curve model',style={'font-size': '17px'}),
                  dcc.RadioItems(
                      id='algorithm_dropdown',
                      options=[{'label': i, 'value': i} for i in ['Micro average of all class', 'Curve of each class']],
                      value='Micro average of all class',
                      labelStyle={'display': 'inline-block'}
                  ), ],
                 style={'width': '48%', 'display': 'inline-block'}),

    ], style={'width': '40%', 'padding': '0 20','margin-left': '300px', 'margin-right': 'auto', 'margin-top': '5px'}),


    html.Div([
        dcc.Graph(id='ME_PR'),
        dcc.Graph(id='ME_ROC'),
    ], style={'width': '60%', 'margin-left': '250px', 'margin-right': 'auto','padding': '0 20'}),

    html.Div([
        html.H1(style={'font-size': '30px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='2. Sentiment Score Trend'),
        html.H1(style={'font-size': '17px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Sentiment score: predict probability of each class'),
        html.H1(style={'font-size': '17px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Trend: average value of total predict/label value (negative as -1, neutral as 0, positive as 1)'),
        html.H1(style={'font-size': '17px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Effective data: Weibo data with more than 50 like/forward/comment'),
        html.H1(style={'font-size': '17px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Sentiment proportion: proportion of each class to total data of the day'),
    ], style={'width': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '0 20'}),

    html.Div([

        html.Div([html.H4('Select Start Date',style={'font-size': '17px'}),
                  dcc.Dropdown(id='startdate_dropdown',
                               options=[{'label': i, 'value': i} for i in date],
                               value='2020-01-10'), ],
                 style={'width': '48%', 'display': 'inline-block'}),

        html.Div([html.H4('Select End Date', style={'font-size': '17px'}),
                  dcc.Dropdown(id='enddate_dropdown',
                               options=[{'label': i, 'value': i} for i in date],
                               value='2020-03-26'), ],
                 style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    ], style={'width': '60%', 'padding': '0 20','margin-left': '250px', 'margin-right': '250px', 'margin-top': '50px'}),

    html.Div([
         dcc.Graph(id='all_sentiment_score'),
         dcc.Graph(id='maxInteract_sentiment_score'),
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20', 'margin-top': '20px'}),

    html.Div([
        dcc.Graph(id='all_sentiment_prob'),
        dcc.Graph(id='maxInteract_sentiment_prob'),
    ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20', 'margin-top': '20px'}),


    html.Div([

        html.Div([html.H4('Select Date', style={'font-size': '20px'}),
                  dcc.Dropdown(id='date_dropdown',
                               options=[{'label': i, 'value': i} for i in date],
                               value='2020-01-10')]),
        dcc.RadioItems(
                id='language_dropdown',
                options=[{'label': i, 'value': i} for i in ['Chinese', 'English']],
                value='Chinese',
                labelStyle={'display': 'inline-block'}
            ),

    ], style={'width': '40%','margin-left': 'auto', 'margin-right': 'auto', 'padding': '0 20','margin-top': '50px'}),


    html.Div([
        html.H1(style={'font-size': '25px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Topics Sentiment Analysis'),
        html.H1(style={'font-size': '17px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Divide Weibo data into 5 topics everyday'),
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20','margin-top': '10px'}),

    html.Div([
        html.H1(style={'font-size': '25px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Keyword Wordcloud'),
        html.H1(style={'font-size': '17px', 'color': '#97A09B', 'margin-left': 'auto', 'margin-right': 'auto',
                       'margin-top': 'auto'},
                children='Key word of total/effective data and each class'),
    ], style={'width': '45%', 'display': 'inline-block', 'padding': '0 20','margin-top': '50px'}),

    html.Div(id='topic_keyword_1', style={'font-size': '13px', 'width': '55%', 'padding': '0 20'}),
    html.Div(id='topic_keyword_2', style={'font-size': '13px', 'width': '55%', 'padding': '0 20'}),
    html.Div(id='topic_keyword_3', style={'font-size': '13px', 'width': '55%', 'padding': '0 20'}),
    html.Div(id='topic_keyword_4', style={'font-size': '13px', 'width': '55%', 'padding': '0 20'}),
    html.Div(id='topic_keyword_5', style={'font-size': '13px', 'width': '55%', 'padding': '0 20','margin-down': '5px'}),

    html.Div([
        dcc.Graph(id='topic_sentiment_analysis'),
    ], style={'width': '35%', 'display': 'inline-block','margin-top': '5px'}),

    html.Div([
        html.Img(id='wordcloud',style={'height':'130%','width':'130%'}),
    ], style={'width': '55%', 'display': 'inline-block','margin-top': '5px'}),



])

@app.callback(
    dash.dependencies.Output('ME_ROC', 'figure'),
    [dash.dependencies.Input('algorithm_dropdown', 'value')])
def update_ME_ROC(algorithm):
    if algorithm=='Micro average of all class':
        return {
            'data': [
                {'x': fpr['micro'], 'y': precision['micro'], 'type': 'line', 'name': 'negative'}
            ],
            'layout': {
                'title': 'micro-averaged ROC over all classes',
                'xaxis': {"title": 'False Positive Rate'},
                'yaxis': {"title": 'True Positive Rate'},
                'height': 300,
                'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
            }
        }

    return {
        'data': [
                    {'x': fpr[0], 'y': tpr[0], 'type': 'line', 'name': 'negative'},
                    {'x': fpr[1], 'y': tpr[1], 'type': 'line', 'name': 'neutral'},
                    {'x': fpr[2], 'y': tpr[2], 'type': 'line', 'name': 'positive'}
        ],
        'layout': {
                    'title': 'ROC of each class',
                    'xaxis': {"title": 'False Positive Rate'},
                    'yaxis': {"title": 'True Positive Rate'},
                    'height': 300,
                    'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
                }
    }

@app.callback(
    dash.dependencies.Output('ME_PR', 'figure'),
    [dash.dependencies.Input('algorithm_dropdown', 'value')])
def update_ME_PR(algorithm):
    if algorithm=='Micro average of all class':
        return {
            'data': [
                {'x': recall['micro'], 'y': precision['micro'], 'type': 'line', 'name': 'negative'}
            ],
            'layout': {
                'title': 'Average precision score, micro-averaged over all classes: AP=' + str(
                    average_precision['micro']),
                'xaxis': {"title": 'Recall'},
                'yaxis': {"title": 'Precision'},
                'height': 300,
                'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
            }
        }


    return {
        'data': [
                    {'x': recall[0], 'y': precision[0], 'type': 'line', 'name': 'negative'},
                    {'x': recall[1], 'y': precision[1], 'type': 'line', 'name': 'neutral'},
                    {'x': recall[2], 'y': precision[2], 'type': 'line', 'name': 'positive'}
        ],
        'layout': {
                    'title': 'precision score of each classes',
                    'xaxis': {"title": 'Recall'},
                    'yaxis': {"title": 'Precision'},
                    'height': 300,
                    'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
                }
    }


@app.callback(
    dash.dependencies.Output('all_sentiment_score', 'figure'),
    [dash.dependencies.Input('startdate_dropdown', 'value'),
     dash.dependencies.Input('enddate_dropdown', 'value')])
def update_all_sentiment_score(startdate, enddate):
    df = data[data['date']>=startdate]
    df = df[df['date']<=enddate]
    avg = df.groupby('date').mean()
    negative = avg['__label__negative'].apply(lambda x: -x)
    positive = avg['__label__positive']
    neutral = avg['__label__neutral']

    avg_score = avg['predict_value']
    date = df['date'].unique()

    return {
        'data': [
                    {'x': date, 'y': negative, 'type': 'line', 'name': 'negative'},
                    {'x': date, 'y': positive, 'type': 'line', 'name': 'positive'},
                    {'x': date, 'y': neutral, 'type': 'line', 'name': 'neutral'},
                    {'x': date, 'y': avg_score, 'type': 'line', 'name': 'trend'}
        ],
        'layout': {
                    'title': 'Sentiment score of all data',
                    'xaxis': {"title": 'date'},
                    'yaxis': {"title": 'sentiment score'},
                    'height': 300,
                    'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
                }
    }

@app.callback(
    dash.dependencies.Output('maxInteract_sentiment_score', 'figure'),
    [dash.dependencies.Input('startdate_dropdown', 'value'),
     dash.dependencies.Input('enddate_dropdown', 'value')])
def update_maxInteract_sentiment_score(startdate, enddate):
    df = data[data['date'] >= startdate]
    df = df[df['date'] <= enddate]
    maxInteract = df[df['max_interact'] > 50]
    max_avg = maxInteract.groupby('date').mean()
    max_negative = max_avg['__label__negative'].apply(lambda x: -x)
    max_positive = max_avg['__label__positive']
    max_neutral = max_avg['__label__neutral']
    max_avg_score = max_avg['predict_value']

    date = df['date'].unique()

    return {
        'data': [
                    {'x': date, 'y': max_negative, 'type': 'line', 'name': 'negative'},
                    {'x': date, 'y': max_positive, 'type': 'line', 'name': 'positive'},
                    {'x': date, 'y': max_neutral, 'type': 'line', 'name': 'neutral'},
                    {'x': date, 'y': max_avg_score, 'type': 'line', 'name': 'trend'}
        ],
        'layout': {
                    'title': 'Sentiment score of effective data',
                    'xaxis': {"title": 'date'},
                    'yaxis': {"title": 'sentiment score'},
                    'height': 300,
                    'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
                }
    }

@app.callback(
    dash.dependencies.Output('all_sentiment_prob', 'figure'),
    [dash.dependencies.Input('startdate_dropdown', 'value'),
     dash.dependencies.Input('enddate_dropdown', 'value')])
def update_all_sentiment_prob(startdate, enddate):
    df = all_proportion_df[all_proportion_df['date']>=startdate]
    df = df[df['date']<=enddate]
    negative_pro = df['negative_pro']
    positive_pro = df['positive_pro']
    neural_pro = df['neural_pro']

    date = df['date'].unique()

    return {
        'data': [
                    {'x': date, 'y': negative_pro, 'type': 'line', 'name': 'negative'},
                    {'x': date, 'y': positive_pro, 'type': 'line', 'name': 'positive'},
                    {'x': date, 'y': neural_pro, 'type': 'line', 'name': 'neutral'}
        ],
        'layout': {
                    'title': 'Sentiment proportion of all data',
                    'xaxis': {"title": 'date'},
                    'yaxis': {"title": 'proportion'},
                    'height': 300,
                    'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
                }
    }


@app.callback(
    dash.dependencies.Output('maxInteract_sentiment_prob', 'figure'),
    [dash.dependencies.Input('startdate_dropdown', 'value'),
     dash.dependencies.Input('enddate_dropdown', 'value')])
def update_maxInteract_sentiment_prob(startdate, enddate):
    df = maxInteract_proportion_df[maxInteract_proportion_df['date']>=startdate]
    df = df[df['date']<=enddate]
    negative_pro = df['negative_pro']
    positive_pro = df['positive_pro']
    neural_pro = df['neural_pro']

    date = df['date'].unique()

    return {
        'data': [
                    {'x': date, 'y': negative_pro, 'type': 'line', 'name': 'negative'},
                    {'x': date, 'y': positive_pro, 'type': 'line', 'name': 'positive'},
                    {'x': date, 'y': neural_pro, 'type': 'line', 'name': 'neutral'}
        ],
        'layout': {
                    'title': 'Sentiment proportion of effective data',
                    'xaxis': {"title": 'date'},
                    'yaxis': {"title": 'proportion'},
                    'height': 300,
                    'margin': {'l': 40, 'b': 70, 'r': 10, 't': 50}
                }
    }

@app.callback(
    Output(component_id='wordcloud', component_property='src'),
    [dash.dependencies.Input('date_dropdown', 'value'),
     dash.dependencies.Input('language_dropdown', 'value')])
def update_wordcloud_ch(date, language):
    if language=='Chinese':
        file_name = 'wordcloud/' + date + '-ch.png'
    else:
        file_name = 'wordcloud/' + date + '-en.png'
    encoded_image = base64.b64encode(open(file_name, 'rb').read()).decode('ascii')
    return 'data:image/png;base64,{}'.format(encoded_image)


@app.callback(
    Output(component_id='topic_keyword_1', component_property='children'),
    [dash.dependencies.Input('date_dropdown', 'value'),
     dash.dependencies.Input('language_dropdown', 'value')])
def update_topic_info(date, language):
    file_name = 'data/topic_result/processed_data' + date + '.csv'
    df = pd.read_csv(file_name)
    if language=='Chinese':
        keyword = df['topic_keyword'].unique()
    else:
        keyword = df['topic_keyword_en'].unique()
    return 'Topic1: '+ keyword[0]

@app.callback(
    Output(component_id='topic_keyword_2', component_property='children'),
    [dash.dependencies.Input('date_dropdown', 'value'),
     dash.dependencies.Input('language_dropdown', 'value')])
def update_topic_info(date, language):
    file_name = 'data/topic_result/processed_data' + date + '.csv'
    df = pd.read_csv(file_name)
    if language=='Chinese':
        keyword = df['topic_keyword'].unique()
    else:
        keyword = df['topic_keyword_en'].unique()
    return 'Topic2: '+keyword[1]

@app.callback(
    Output(component_id='topic_keyword_3', component_property='children'),
    [dash.dependencies.Input('date_dropdown', 'value'),
     dash.dependencies.Input('language_dropdown', 'value')])
def update_topic_info(date, language):
    file_name = 'data/topic_result/processed_data' + date + '.csv'
    df = pd.read_csv(file_name)
    if language=='Chinese':
        keyword = df['topic_keyword'].unique()
    else:
        keyword = df['topic_keyword_en'].unique()
    return 'Topic3: '+keyword[2]

@app.callback(
    Output(component_id='topic_keyword_4', component_property='children'),
    [dash.dependencies.Input('date_dropdown', 'value'),
     dash.dependencies.Input('language_dropdown', 'value')])
def update_topic_info(date, language):
    file_name = 'data/topic_result/processed_data' + date + '.csv'
    df = pd.read_csv(file_name)
    if language=='Chinese':
        keyword = df['topic_keyword'].unique()
    else:
        keyword = df['topic_keyword_en'].unique()
    return 'Topic4: '+keyword[3]

@app.callback(
    Output(component_id='topic_keyword_5', component_property='children'),
    [dash.dependencies.Input('date_dropdown', 'value'),
     dash.dependencies.Input('language_dropdown', 'value')])
def update_topic_info(date, language):
    file_name = 'data/topic_result/processed_data' + date + '.csv'
    df = pd.read_csv(file_name)
    if language=='Chinese':
        keyword = df['topic_keyword'].unique()
    else:
        keyword = df['topic_keyword_en'].unique()
    return 'Topic5: '+keyword[4]


@app.callback(
    dash.dependencies.Output('topic_sentiment_analysis', 'figure'),
    [dash.dependencies.Input('date_dropdown', 'value')])
def update_topic_sentiment_analysis(date):
    file_name = 'data/topic_result/processed_data'+date+'.csv'
    df = pd.read_csv(file_name)
    df = df.sort_values(by=['topic'], ascending=False)
    topic = df['topic'].unique()
    count = df.groupby('topic').count()['id']
    avg = df.groupby('topic').mean()
    negative = avg['__label__negative']*200
    positive = avg['__label__positive']*200
    neutral = avg['__label__neutral']*200

    return {
        'data': [dict(x=topic, y=negative, type='bar', name='negative score'),
                 dict(x=topic, y=positive, type='bar', name='positive score'),
                 dict(x=topic, y=neutral, type='bar', name='neutral score'),
                 dict(x=topic, y=count, type='line', name='topic count', marker={'color': '1C9CE5'})],
        'layout': {
                'title': 'Topic Sentiment Analysis',
                'xaxis': {"title": 'topic'},
                'yaxis': {"title": 'sentiment score'},
                # 'yaxis2': {"title": 'topic count','side':'right','overlaying':'y'},
                'height': 480,
                'margin': {'l': 50, 'b': 50, 'r': 20, 't': 30}
         }
    }



if __name__ == '__main__':
    app.run_server(debug=True, port=8050)


