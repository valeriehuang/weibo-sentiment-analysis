import re
import sys

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
import urllib.parse
from selenium import webdriver
import datetime
import time as systime
from selenium.webdriver.firefox.webdriver import FirefoxProfile
import unicodecsv as csv

base_url = 'http://s.weibo.com/weibo/'

def crawl(query_file):
    with open(query_file) as f:
        query = f.readlines()
    query = [x.strip() for x in query]
    # print urllib.quote(urllib.quote(each_query[0]))
    for line in query:
        s = line.split('\t')
        keyword = s[0]  # urllib.quote(urllib.quote(s[0]))
        date = s[1]
        start = s[2]
        end = s[3]
        page = s[4]
        # create csv file for each query
        out_file = 'data' + date + '.csv'
        with open(out_file, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(["id", "keyword", "date", "content", "time", "forward", "comment", "like"])
        csvfile.close()
        scrap_each_query(keyword, date, start, end, page, out_file)


def check_forward(content):
    for c in content:
        p_txt = c.find_all('p', class_='txt')[0].text
        if (u"转发微博" in p_txt) or p_txt == '':
            return True
    return False


def get_four_stats(one_card_act):
    # function to get statistics for: forward转发, comment评论, like点赞
    first = one_card_act.ul.find_all("li")
    res = []
    for tag in first:
        content = tag.find("a").contents

        if len(content) == 1:
            content = str(content[0])
            l = [int(s) for s in content.split() if s.isdigit()]
            if len(l) != 0:
                res.append(l[0])
            else:
                res.append(0)
        elif len(content) == 3:
            content = str(content[2])
            l = re.findall(r"\d+", content)
            if len(l) != 0:
                res.append(int(l[0]))
            else:
                res.append(0)
    return res[0], res[1], res[2], res[3]


def scrap_each_query(keyword, date, start, end, page, out_file):
    real_keyword = keyword
    keyword = urllib.parse.quote(urllib.parse.quote(keyword))
    profile = FirefoxProfile("/Users/Rock_Wang/Library/Application Support/Firefox/Profiles/038xacgn.default-release")
    driver = webdriver.Firefox(profile)
    url = base_url + keyword + "&typeall=1&suball=1&timescope=custom:" + start + ":" + end + "&page=" + "1"
    driver.get(url)
    systime.sleep(5)
    count = 0
    print("crawling data on {} and using keyword {}".format(date, keyword))
    for i in range(int(page)):
        url = base_url + keyword + "&typeall=1&suball=1&timescope=custom:" + start + ":" + end + "&page=" + str(i + 1)
        driver.get(url)
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")
        for card in soup.find_all('div', class_='card'):
            content = card.find_all('div', class_='content')
            card_act = card.find_all('div', class_='card-act')
            # check if forward
            if check_forward(content):
                continue

            for c in content:
                p_txt = c.find_all('p', class_='txt')[0].text
                txt = p_txt.replace('\n', '').strip()
                # print(txt)
                p_time = c.find_all('p', class_='from')[0]
                time = p_time.find('a').contents[0]
                time = time.replace('\n', '').strip()
                # print(time)
            for act in card_act:
                keep, forward, comment, like = get_four_stats(act)
                # print("{} {} {}".format(forward, comment, like))

            # print(count)
            count = count + 1

            with open(out_file, 'ab+') as write_obj:
                # Create a writer object from csv module
                csv_writer = csv.writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow([count, real_keyword, date, txt, time, forward, comment, like])
        print("Finished writing page {}".format(i))
    driver.close()
    write_obj.close()


def get_four_stats(one_card_act):
    # function to get statistics for: forward转发, comment评论, like点赞
    first = one_card_act.ul.find_all("li")
    res = []
    for tag in first:
        content = tag.find("a").contents

        if len(content) == 1:
            content = str(content[0])
            l = [int(s) for s in content.split() if s.isdigit()]
            if len(l) != 0:
                res.append(l[0])
            else:
                res.append(0)
        elif len(content) == 3:
            content = str(content[2])
            l = re.findall(r"\d+", content)
            if len(l) != 0:
                res.append(int(l[0]))
            else:
                res.append(0)
    return res[0], res[1], res[2], res[3]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid argument")
        print("Usage:\ncrawler.py [query_file]")
        exit(0)
    query_file = sys.argv[1]
    crawl(query_file)
