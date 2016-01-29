# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:54:06 2016

@author: brian
"""
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import *

from sklearn import svm
from sklearn.metrics import accuracy_score

def unlist(element):
    return ''.join(element)

#reading json
def read_scraped_jason(filename):
    df = pd.read_json(filename)
    
    for column in df.columns:
        df[column] = df[column].apply(unlist)
    # gets only first 10 characters of date: year/month/day
    df['date'] = df['date'].apply(lambda x: x[:10])
    df['date'] = pd.to_datetime(df['date'])
    
    # if any removes duplicate posts
    df = df.drop_duplicates(subset = ['keywords'])
    # sorts dataframe by post date
    df = df.sort_values(by='date')
    df['text'] = df['keywords'] + df['body'] 
 
    df = df.drop('body', 1)
    df = df.drop('keywords', 1)
    
    df = df.dropna()
    
    return df

def clean_text(text):
    text = text.lower()
    
    text = BeautifulSoup(text).get_text()
    
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    
    stop = stopwords.words('english')
    clean = [word for word in text if word not in stop]
    
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in clean]
    
    return stemmed


###############################################################################
###############################################################################   
###############################################################################       

df = read_scraped_jason('busweek/items.json')
df['text'] = df['text'].apply(clean_text)


market_df = pd.read_csv('sp_500_2014.csv', index_col="Date",
                        parse_dates=True, usecols=['Date','Open','Close'],
                        na_values=['nan'])
                        
market_df = market_df.drop(pd.to_datetime('2014-12-24'))

#create a list of the words we'll search for
#start by counting the occurneces of all words then choose the top 500
word_dict = dict()
for index, row in df.iterrows():
    for word in row.loc['text']:
        if not unicode.isdigit(word):
            if word_dict.has_key(word):
                word_dict[word] = word_dict[word] + 1;
            else:
                word_dict[word] = 1;
                
sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
features = sorted_words[0:10000]
features_dict = {}
for i in range(len(features)):
    features_dict[features[i]] = i;

#combine all the article contents into their appropriate day
data_set = pd.DataFrame(index=market_df.index, columns=['text'])

for i, row in df.iterrows():
    
    if df.loc[i, 'date'] in data_set.index:
        if pd.isnull(data_set.loc[df.loc[i, 'date']]).any():
            data_set.loc[df.loc[i, 'date'], 'text'] = list(df.loc[i, 'text'])
        else:
            data_set.loc[df.loc[i, 'date'], 'text'] += df.loc[i, 'text']

#go through each day and see which of the 500 chosen words are present
feature_df = pd.DataFrame(index=data_set.index, columns=[range(10000)])
feature_df = feature_df.fillna(0)

for i, row in data_set.iterrows():
    for word in row['text']:
        if word in features:
            feature_df.loc[i, features_dict[word]] = 1;

#create the output from the market data
#if close > open, output = 1
output_df = pd.DataFrame(index=data_set.index, columns=['rising'])
for i, row in market_df.iterrows():
    if row['Close'] > row['Open']:
        output_df.loc[i, 'rising'] = 1
    else:
        output_df.loc[i, 'rising'] = 0
            
features_train = feature_df.sample(n = 150)
train_ind = features_train.index

helper_df = feature_df.drop(train_ind)
features_validate = helper_df.sample(n = 50)
validate_ind = features_validate.index

features_test = helper_df.drop(validate_ind)
test_ind = features_test.index

label_train = output_df.ix[train_ind]
label_validate = output_df.ix[validate_ind]
label_test = output_df.ix[test_ind]


features_train = features_train.as_matrix()
features_test = features_test.as_matrix()
features_validate = features_validate.as_matrix()

label_train = list(label_train.values.flatten())
label_test = list(label_test.values.flatten())
label_validate = list(label_validate.values.flatten())

C_values= [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

print "Trying SVM rbf now"
best_accuracy = 0

for curr_C in C_values:
                
    clf = svm.SVC(kernel = 'rbf', C = curr_C)
    clf.fit(features_train, label_train)
    
    pred = clf.predict(features_validate)
    curr_accuracy = accuracy_score(label_validate, pred)
    
    print "C = %f; accuracy: %f" %  (curr_C, curr_accuracy)
          
    if best_accuracy < curr_accuracy:
        best_accuracy = curr_accuracy
        best_C = curr_C
        
print "Best C = %f; Best Accuracy = %f" % (best_C, curr_accuracy)

clf = svm.SVC(kernel = 'rbf', C = best_C)
clf.fit(features_train, label_train)

pred = clf.predict(features_test)
accuracy = accuracy_score(label_test, pred)

print "Final test accuracy: %f" %  (accuracy)

            
#def main():
#    df = read_scraped_jason('busweek/items.json')
#  
#if __name__ == "__main__":
#    main()