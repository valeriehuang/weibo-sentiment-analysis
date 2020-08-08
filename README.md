***Weibo Hot Search Topic Analysis  
&  
Weibo Content Sentiment Analysis***

All related data has been uploaded to this repository (including the labeled data), and can be used for future projects.

Our Visualization result can be accessed by the following URLS  
Result of Hot Search Topics  
http://ec2-18-218-241-59.us-east-2.compute.amazonaws.com:8051/  
Result of Weibo Sentiment Analysis  
http://ec2-18-218-241-59.us-east-2.compute.amazonaws.com:8050/  

All data needed for this project is in  
`cmpt733/scripts/All_you_need.zip`  
`cmpt733/Weibo_unlabed_and_labeled.zip`  
`cmpt733/scripts/for_wordCloud_use.zip`  
Unzip them to get start, do not change their location

1. Hot Search Topic
    * Data Collection  
    `>python weibo_hot_topics_scraping.py`
    * Topic Analysis  
    `hot_topic_analysis.ipynb`
    * Result Visualization  
    `>python hot_search_plot.py`  
    URL for web frontend: http://localhost:8051/  
2. Weibo Content Sentiment Analysis
    * Data Collection  
    Edit `query.txt` as you need, crawler will take this file as input  
    `>python crawler.py`  
    * Data Pre-process & Topic aggregation  
    `>python split_data.py`  
    `>python pre-process.py [input_directory] [output_directory]`  
    `>python topic_analysis.py` 
    * Model training, testing  
    `>python train_test.py [train_directory] [test_directory]`
    * Model Evaluation & Test result Visualization  
    `>python plot_sentiment_analysis.py`  
    URL for web frontend: http://localhost:8050/
    * WordCloud Generation  
    `>python wordcloud.ipynb`  
    * Translation  
    `>python translation.ipynb`  
