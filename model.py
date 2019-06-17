#IMPORTING DEPENDENCIES
import numpy as np
import tensorflow as tf
import csv
from tensorflow import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import pickle 
import nltk
###############################################################################################################################
#READING CSV INTO X_TRAIN AND Y_TRAIN
def read_file(filename):
    X_train=[]
    Y_train=[]
    with open(filename, 'r') as csvfile: 
        csvreader = csv.reader(csvfile) 
        for row1,row2 in csvreader: 
            Y_train.append(int(row1))
            X_train.append(row2)
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    number_of_output_neurons=Y_train[np.argmax(Y_train)]+1
    Y_train_oh=to_categorical(Y_train,number_of_output_neurons)
    return X_train,Y_train,Y_train_oh,number_of_output_neurons
###############################################################################################################################
#READING GLOVE FILE
word_to_index={}
index_to_word={}
word_to_vec_map={}
fp= open('word_to_index.pkl', 'rb')
word_to_index = pickle.load(fp)
fp= open('index_to_word.pkl', 'rb')
index_to_word = pickle.load(fp)
fp= open('word_to_vec_map.pkl', 'rb')
word_to_vec_map = pickle.load(fp)
fp.close()
###############################################################################################################################
#SENTENCE TO INDICES
def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))
    print(X_indices.shape)
    for i in range(m):
        X[i]=X[i].lower()
        sentence_words=nltk.TreebankWordTokenizer().tokenize(X[i])
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j+1
    return X_indices
###############################################################################################################################
#Making the embedding layer
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_matrix = np.zeros((vocab_len,emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer
###############################################################################################################################
#MAKING THE MODEL :)
def RNN_LSTM(input_shape, word_to_vec_map, word_to_index,number_of_output_neurons):
    sentence_indices = Input(input_shape,dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)   
    X = Bidirectional(LSTM(200,return_sequences=True))(embeddings)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Bidirectional(LSTM(200,return_sequences=False))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.3)(X)
    X = Dense(number_of_output_neurons)(X)
    X = Activation('softmax')(X)
    model = Model(inputs=sentence_indices,outputs=X)
    return model
###############################################################################################################################
#Model instance,Compiling,training,etc
maxLen = 20
number_of_models=1
for i in range(number_of_models):
    filename='train'+str(i)+'.csv'
    X_train,Y_train,Y_train_oh,number_of_output_neurons=read_file(filename)
    model = RNN_LSTM((maxLen,), word_to_vec_map, word_to_index,number_of_output_neurons)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    X_train_indices =sentences_to_indices(X_train, word_to_index, maxLen)
    model.fit(X_train_indices, Y_train_oh, epochs = 40, batch_size = 32, shuffle=True)
    #Saving model weights and architechture 
    weights_save='model_weights'+str(i)+'.h5'
    archtecture_save='model_architecture'+str(i)+'.json'
    model.save_weights(weights_save)
    with open(archtecture_save, 'w') as f:
        f.write(model.to_json())
