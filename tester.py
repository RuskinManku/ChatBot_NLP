#####################################################################################################################
import numpy as np
import tensorflow as tf
import csv
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.models import model_from_json
import pickle 
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 
import os
#from aion.util.spell_check import SpellCorrector
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#####################################################################################################################
def index_to_answers(answers_file):
    index_to_answer={}
    with open(answers_file,'r') as file:
        csvreader = csv.reader(file) 
        for row1,row2 in csvreader: 
            #print(str(row1) + ':' + str(row2))
            index_to_answer[int(row1)]=row2
    return index_to_answer

maxLen=20
#####################################################################################################################
word_to_index={}
index_to_word={}
word_to_vec_map={}
fp= open('word_to_index.pkl', 'rb')
word_to_index = pickle.load(fp)
#spell_corrector = SpellCorrector(dictionary=word_to_index)
fp= open('index_to_word.pkl', 'rb')
index_to_word = pickle.load(fp)
fp= open('word_to_vec_map.pkl', 'rb')
word_to_vec_map = pickle.load(fp)
fp.close()
#####################################################################################################################
def sentences_to_indices(X, word_to_index, max_len,remove_stop):
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))
    for i in range(m):
        X[i]=X[i].lower()
        sentence_words=nltk.TreebankWordTokenizer().tokenize(X[i])
        if remove_stop==1:
            new_sentence=[]
            for word in sentence_words:
                if word not in stop_words:
                    new_sentence.append(word)
            sentence_words=new_sentence
        j = 0
        print(sentence_words)
        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
            else:
                print(w)
                #X_indices[i,j]=0
            j = j+1
    return X_indices
#####################################################################################################################
models=[]
number_of_models=2
for i in range(number_of_models):
    weight_name='model_weights'+str(i)+'.h5'
    architecture_name='model_architecture'+str(i)+'.json'
    with open(architecture_name, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(weight_name)
    models.append(model)
#####################################################################################################################
def removeprodname(sentence):
    sentence=sentence.lower()
    sentence_words=nltk.TreebankWordTokenizer().tokenize(sentence)
    number_of_words=len(sentence_words)
    for i in range(number_of_words):
        if sentence_words[i]=='growth' and sentence_words[i+1]=='plus':
            sentence_words[i]='it'
            sentence_words[i+1]=''
        if sentence_words[i]=='eno':
            sentence_words[i]='it'
        if sentence_words[i]=='protein' and sentence_words[i+1]=='plus':
            sentence_words[i]='it'
        if sentence_words[i]=='crocin':
            sentence_words[i]='it'
        if sentence_words[i]=='otrivin':
            sentence_words[i]='it'
        if sentence_words[i]=='horlicks':
            sentence_words[i]='it'
        if sentence_words[i]=='brush':
            sentence_words[i]='it'
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='base':
            sentence_words[i]='it'
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='rapid':
            sentence_words[i]='it'
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='repair':
            sentence_words[i]='it'
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='herbal':
            sentence_words[i]='it'
    sentence_words=' '.join(sentence_words)
    return sentence_words
#####################################################################################################################
def check_product(sentence):
    sentence=sentence.lower()
    sentence_words=nltk.TreebankWordTokenizer().tokenize(sentence)
    number_of_words=len(sentence_words)
    for i in range(number_of_words):
        if sentence_words[i]=='growth' and sentence_words[i+1]=='plus':
            return 0
        if sentence_words[i]=='eno': #and sentence_words[i+1] != 'cooling':
            return 1
        if sentence_words[i]=='protein' and sentence_words[i+1]=='plus':
            return 2
        if sentence_words[i]=='crocin':
            return 3
        if sentence_words[i]=='otrivin':
            return 4
        if sentence_words[i]=='horlicks':
            return 5
        if sentence_words[i]=='brush':
            return 6
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='base':
            return 7
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='rapid':
            return 8
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='repair':
            return 9
        if sentence_words[i]=='sensodyne' and sentence_words[i+1]=='herbal':
            return 10
    return 11
#####################################################################################################################
#index_to_sentence=index_to_answers('answers.csv')
while [1]:
    print ("Type your question ")
    question=input("Enter your question :")
    product_number=check_product(question)
    print(product_number)
    question_modify=removeprodname(question)
    print(question_modify)
    if product_number<=10:
        #answers_file='answers'+str(product_number)+'.csv'
        #index_to_answer=index_to_answers(answers_file)
        x_test_typezero=np.array([question_modify])
        x_test_typeone=np.array([question])
        X_test_indices_zero = sentences_to_indices(x_test_typezero, word_to_index, maxLen,0) #ALSO CHECKS FOR SPELLINGS 
        X_test_indices_one = sentences_to_indices(x_test_typeone, word_to_index, maxLen,1) #ALSO CHECKS FOR SPELLINGS 
        predictions=models[0].predict(X_test_indices_zero)
        index_zero=np.argmax(predictions)
        confidence_zero=predictions[:,index_zero]*100
        predictions=models[1].predict(X_test_indices_one)
        index_one=np.argmax(predictions)
        confidence_one=predictions[:,index_one]*100
        print('Confidence in file zero:'+str(confidence_zero)+'  for index:'+str(index_zero))
        print('Confidence in file one:'+str(confidence_one)+'  for index:'+str(index_one))
    else:
        x_test=np.array([question])
        X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen,1)
        predictions=models[1].predict(X_test_indices)
        index=np.argmax(predictions)
        confidence=predictions[:,index]*100
        print('No catch for product name')
        print('Catch in type 1 file index:'+str(index)+'  with confidence:'+str(confidence))
#####################################################################################################################