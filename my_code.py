from concurrent.futures import process
import gzip
import random
from tracemalloc import stop
import nltk
import string
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk import ngrams
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
#nltk.download('stopwords')

punc = '''!()-[]{;}:'"\,<>./?@#$%^&*_~'''
sent_list = []
def sample_lines(file_name,lines):
    file = gzip.open(file_name,"rb")
    lines_list = list(file)
    sample_list = []
    for line in lines_list[:lines]:
        sample_list.append(line)
    
    return random.sample(sample_list,lines)

def process_sentences(sampled_lines):
    for line in sampled_lines:
        final_list = [] # list of lists with each sentence as a list of words with tags
        sentence_tokens = nltk.word_tokenize(str(line))
        word_tag = nltk.pos_tag(sentence_tokens)
        lower = [(word.lower(),tag) for word,tag in word_tag]
        for item in lower:
            if item[0] not in stopwords.words('english'):
                if item[0] not in punc:
                    final_list.append(item)
        sent_list.append(final_list)        
    return sent_list

def create_samples(processed_sentences,limit):
    five_gram_list = []
    for sentence in processed_sentences[:limit]:
        five_grams = list(ngrams(sentence,5))
        if five_grams!= []:
            five_gram_list.append(random.choice(five_grams))
        
    #creating feature list
    features = []
    for gram in five_gram_list:
        feat_1 = (gram[0][0][-2:]) + "_"+str(1)
        feat_2 = (gram[1][0][-2:]) + "_"+str(2)
        feat_4 = (gram[3][0][-2:]) + "_"+str(4)
        feat_5 = (gram[4][0][-2:]) + "_"+str(5)
        
        if gram[2][1] == 'VBN':
            verb_value =  1
        else:
            verb_value = 0
        feature_tuple = ((feat_1,feat_2,feat_4,feat_5),verb_value)
        features.append(feature_tuple)

    return features
    

def create_df(features):
    #creating vectors
    ending_list = []
    create_df.verb_values_list = []
    for sent_features in features:
        for gram in sent_features[0]:
            ending_list.append(gram)
        create_df.verb_values_list.append(sent_features[1])    
    
    #creating vectors
    feature_vectors = []
    for feature in features:
        vector_list = []
        for ending in ending_list:
            if ending in feature[0]:
                vector_list.append(1)
            else:
                vector_list.append(0)
        feature_vectors.append(vector_list)
    # print(np.array(feature_vectors))
    df = pd.DataFrame(data = feature_vectors,columns=ending_list)
    return df

def split_samples(df,test_percent):
    X = df.reset_index(drop=True)
    y = create_df.verb_values_list
    train_test_samples = train_test_split(X,y, test_size = test_percent/100)
    return(train_test_samples)

def train(X,y,kernel): 
    if kernel == 'rbf':
        clf = svm.SVC(kernel='rbf')
    else: 
        clf = svm.SVC(kernel='linear')
    model = clf.fit(X,y)
    return model

def eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    precision_score = "Precision:"+ str(precision)
    recall_score = "Recall:"+str(recall)
    f_measure = "F-measure:" + str((2*precision*recall)/(precision+recall))
    return precision_score, recall_score, f_measure
