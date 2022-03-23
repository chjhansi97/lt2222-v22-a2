import gzip
import random
import math
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.util import ngrams
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from sklearn.svm import SVC
from sklearn import metrics
import string

def sample_lines(file_name, lines=100000):
    file = gzip.open(file_name,"rb")
    lines_list = list(file)
    sample_list = []
    for line in lines_list[:lines]:
        sample_list.append(line.decode('utf8').strip())
    
    return random.sample(sample_list,lines)

def process_sentences(sampled_lines):
    sent_list = []
    for line in sampled_lines:
        final_list = [] # list of lists with each sentence as a list of words with tags
        sentence_tokens = nltk.word_tokenize(line)
        word_tag = nltk.pos_tag(sentence_tokens)
        lower = [(word.lower(),tag) for word,tag in word_tag]
        for item in lower:
            if item[0] not in stopwords.words('english'):
                if item[0] not in string.punctuation:
                    final_list.append(item)
        sent_list.append(final_list)        
    return sent_list

def create_samples(processed_sentences, limit=50000):
    five_gram_list = []
    for sentence in processed_sentences:
        five_grams = list(ngrams(sentence,5))
        for gram in five_grams:
            five_gram_list.append(gram)

    random_five_gram = random.randint(0, (len(five_gram_list)-limit))
    limit = five_gram_list[random_five_gram:random_five_gram+limit]
    #creating feature list
    samples = []
    for gram in limit:
        feat_1 = (gram[0][0][-2:]) + "_"+str(1)
        feat_2 = (gram[1][0][-2:]) + "_"+str(2)
        feat_4 = (gram[2][0][-2:]) + "_"+str(3)
        feat_5 = (gram[3][0][-2:]) + "_"+str(5)
        
        if gram[2][1] == 'VBN':
            verb_value =  1
        else:
            verb_value = 0
        sample = (feat_1,feat_2,feat_4,feat_5,verb_value)
        samples.append(sample)

    return samples

def create_df(samples):
    columns = []
    rows = []
    for sample in samples:
        features = sample[:4]
        for feature in features:
            if feature not in columns:
                columns.append(feature)
    columns.append('VBN')
    
    for sample in samples:
        features = sample[:4]
        verb = sample[4]
        row_vector = []
        for feature in columns:
            if (feature != 'VBN') and (feature in features):
                row_vector.append(1)
            elif (feature != 'VBN') and (feature not in features):
                row_vector.append(0)
        row_vector.append(verb)
        rows.append(row_vector)
    df = pd.DataFrame(np.array(rows),columns=columns)
    return df

def split_samples(df, test_percent=20):
    X = df.iloc[:,0:len(df.columns)-2]
    y = df.iloc[:,len(df.columns)-1]
    percent = round(len(df) * (test_percent/100))
    X_train, y_train, X_test, y_test = X[percent:], y[percent:], X[:percent], y[:percent]
    
    return(X_train, y_train, X_test, y_test)

def train(X_train, y_train, kernel='linear'):
    if kernel == 'rbf':
        clf = SVC(kernel='rbf')
    elif kernel == 'linear': 
        clf = SVC(kernel='linear')
    else:
        return
    model = clf.fit(X_train, y_train)
    return model

def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f_measure = metrics.f1_score(y_test, y_pred)
    precision_score = "Precision:"+ str(precision)
    recall_score = "Recall:"+str(recall)
    f_measure_score = "F-measure:" + str(f_measure)
    print(model,"\n", precision_score, "\n",recall_score, "\n",f_measure_score)
