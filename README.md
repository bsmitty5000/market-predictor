# market-predictor
Using Scrapy and SVM for Stock Market prediction

My first attemp at a machine learning algorithm from scratch. 

I borrowed heavily from http://francescopochetti.com/scrapying-around-web/
for the Scrapy code, with some modifications, and the general idea.

The basic idea was to use machine learning to see if news article content
correlates with Stock Market performance. Instead of using sentiment analysis
on the news articles, like Francesco did in the above link, I used the basic
idea of spam filters to try and correlate.

The feature set in the first commit is the 10,000 most frequently occuring
words in the news articles from 2014. The label set is a True/False indicating
if the Stock Market rose on that particular day. My thinking is this is a more
organic way at doing what sentiment analysis does, since I'm using the behavior
of the market to decide which words might correctly correlate.

