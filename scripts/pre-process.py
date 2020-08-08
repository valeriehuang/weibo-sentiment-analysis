import sys
from weibo_preprocess_toolkit import WeiboPreprocess
import pandas as pd
import re
import cn2an
import numpy as np
import datetime
import os


# "转发微博","匹配：转发微博"
# "来自.{1,20} *?$","匹配 来自***客户端"
# "([^\s]+?的){0,1}秒拍视频","匹配 秒拍视频"
# "(\d{2,4}年){0,1}\d+月\d+日","匹配 几月几日"
# "\d{1,4}年","匹配 年"
# "\d{1,2}日","匹配 日"
# "网页链接","匹配 网页链接"
# "分享图片","匹配 分享图片"
# "最新博文","匹配 最新博文"
# "赞\[\d+\]","匹配 赞[]"
# "转发\[\d+\]","匹配 转发[]"
# "收藏\[\d+\]","匹配 收藏[]"
# "评论\[\d+\]","匹配 评论[]"
# "\[超话\]","匹配：[超话]"
# "收藏\d+月\d+日 \d+:\d+\s?","匹配： 收藏09月11日 18:57 "
# "#[^\s:：，,。@#]+? {0,3}· {0,3}[^\s:：，,。@#]+?\[地点\]#","匹配地点： #南京·大行宫[地点]#"
# "\s+?[^\s:：，,。@#]+? {0,3}· {0,3}[^\s:：，,。@#]+?(\s+|$)","匹配微博末尾的 地点"
# "@[^\s:：，,。@]{1,20}","匹配 @某某人"
# "\d{1,2}([:：]|点|分)\d{1,2}(分|秒){0,1}","匹配 几比几或者 几点几分，几时几秒"
# "\d+\.\d+w{0,1}万{0,1}","匹配 小数"
# "\d+(\.\d+){0,1}%","匹配 百分数"
# "(详见){0,1}\.{1,6}(展开){0,1}全文c{0,1}","匹配 详见...展开全文c"
# "(amp|;quot|;gt|;lt)","匹配 html special chars"
# "(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]","匹配 url"
# "[\w!#$%&'*+/=?^_`{|}~-]+(?:\.[\w!#$%&'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?","匹配 email"

# covered case in replace_num(str)
# X年, X月, X日, X小时, X时, X分钟/分, X天, X秒, follow alphabet, X:X

def replace_num(str):
    if isinstance(str, float):
        return ""
    start = -1
    new_str = ""
    for i in range(len(str) - 1):
        # print(str[i])
        # print(str[i].isnumeric())
        if str[i].isdigit():
            # print(str[i])
            if start != -1 and str[i + 1].isdigit():
                continue
            if start == -1:
                start = i
            if start != -1 and str[i + 1].isdigit() is False:
                try:
                    num = cn2an.an2cn(str[start:i + 1], 'low')
                except ValueError:
                    num = str[start:i + 1]
                start = -1
                new_str = new_str + num
            # if str[i + 1] != '年' and str[i + 1] != '月' and str[i + 1] != '日' and str[i + 1] != '天' and str[i] != ':' and \
            #         str[i + 1] != '小' and str[i + 1] != '时' and str[i + 1] != '分' and str[i + 1] != '秒' and \
            #         str[i + 1].isalpha() is False and str[i + 1].isdigit() is False and start != -1:
            #     print(str[start:i+1])
            #     num = cn2an.an2cn(str[start:i+1], 'low')
            #
            #     start = -1
            #     new_str = new_str + num
        else:
            new_str = new_str + str[i]
    return new_str


def trim_end(str):
    if str[-2:] == '全文' and str[-5:-3] == '展开':
        return str[:-5]
    else:
        return str


def extract_date_time(date_time):
    # "(\d{2,4}年){0,1}\d+月\d+日","匹配 几月几日"
    # "\d{1,4}年","匹配 年"
    # "\d{1,2}日","匹配 日"
    year_regex = '\d{1,4}年'
    month_day_regex = '\d+月\d+日'
    time_regex = '\d+:\d+'
    year = re.findall(year_regex, date_time)
    month_day = re.findall(month_day_regex, date_time)
    time = re.findall(time_regex, date_time)
    try:
        month = int(month_day[0][0:2])
        day = int(month_day[0][3:5])
        h = int(time[0][0:2])
        m = int(time[0][3:5])
    except IndexError:
        # TODO: need further modification on date extraction
        month = 1
        day = 1
        h = 0
        m = 0
    # only care about year of 2020
    date_time = datetime.datetime(year=2020, month=month, day=day, hour=h, minute=m)
    return date_time


def fill_label(num_label):
    if int(num_label) == 0:
        return '__label__negative'
    if int(num_label) == 1:
        return '__label__positive'
    if int(num_label) == 2:
        return '__label__neutral'
    else:
        return num_label


# clean and label data
def process_file(in_file, out_file):
    df = pd.read_csv(in_file)
    out_df = df[['id', 'label', 'keyword', 'date', 'content', 'time', 'forward', 'comment', 'like']].copy()
    # print(df['content'])
    # replace numeric numbers with Chinese characters
    out_df['content'] = out_df['content'].apply(replace_num)
    out_df['content'] = out_df['content'].apply(tool.preprocess)
    out_df['content'].replace('', np.nan, inplace=True)
    out_df = out_df.dropna(axis=0, how='any')
    # get ride of "展开全文" in the end of the content
    out_df['content'] = out_df['content'].apply(trim_end)
    # format date
    out_df['time'] = out_df['time'].apply(extract_date_time)
    out_df['label'].fillna(0, inplace=True)
    # filling word labels
    out_df['label'] = out_df['label'].apply(fill_label)
    # print(out_df['label'])
    out_df.to_csv(out_file, index=False)


def get_file_names(input_directory):
    result_list = []
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):
            result_list.append(file_name)
    return result_list


def process_data_by_file(input_directory, output_directory):
    out_data = pd.DataFrame()
    file_list = get_file_names(input_directory)
    for file_name in file_list:
        path = input_directory + "/" + file_name
        out_file_name = 'processed_' + file_name
        out_path = output_directory + '/' + out_file_name
        process_file(path, out_path)
        # out_data = pd.concat([out_data, data], ignore_index=True)
    return out_data


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Invalid argument")
        print("Usage:\npre-process.py [input_directory] [output_directory]")
        exit(0)
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    tool = WeiboPreprocess()
    process_data_by_file(input_directory, output_directory)
