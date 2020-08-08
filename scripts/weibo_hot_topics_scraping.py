import os
import time
import requests
import json
import pandas as pd

def requests_web_data(url):
    try:
        headers = {"User-Agent": "", "Cookie": ""}
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
    except:
        print('requests error!')
    else:
        return r.content
 
def get_weibo_historical_data():
    topic_df = pd.DataFrame(columns=['content', 'end_time', 'start_time', 'count'])
    
    latest_time_id_url = 'https://www.eecso.com/test/weibo/apis/getlatest.php'
    latest_time_id = json.loads(requests_web_data(latest_time_id_url).decode('utf-8'))[0] 
     
    # get time_id
    time_ids = []
    for x in range(48438, int(latest_time_id) + 1, 180):    # time_id=48438：2020-01-01
        time_id_url = 'https://www.eecso.com/test/weibo/apis/getlatest.php?timeid=' + str(x)
        time_data = json.loads(requests_web_data(time_id_url).decode('utf-8'))
        if time_data is not None:
            time = time_data[1].split(' ')[1].split(':')[0]
            if time == '00' or time == '12':
                time_ids.append(time_data[0])
    if time_ids[-1] != latest_time_id:
        time_ids.append(latest_time_id)
        
    # get hot topic from time_id
    weibo_hot_data = []
    for time_id in time_ids:
        print(time_id)
        historical_data_url = 'https://www.eecso.com/test/weibo/apis/currentitems.php?timeid=' + str(time_id)
        data = json.loads(requests_web_data(historical_data_url).decode('utf-8'))
        for item in data:
            item = pd.Series(item, index = topic_df.columns)
            topic_df = topic_df.append(item, ignore_index=True) 
    print(topic_df)   
    topic_df.to_csv('weibo_hot_topic.csv')

get_weibo_historical_data()
