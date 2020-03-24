# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:59:40 2020

@author: admin
"""
#import NLP packages
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
import random

#import deep learning framework packages(keras)
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

#intializing variabels
words = []
classes = []
documents = []
ignore_words = ['?','!']

#open and load json file
data_file = open('intents.json').read()
intents = json.loads(data_file)


#preprocessing

#looping through json file  
for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        #tokenize all the words
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #adding documents
        documents.append((word, intent['tag']))
        
        #adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
#lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words]
words = sorted(list(set(words)))   

#sort classes
classes = sorted(list(set(classes))) 
#documents is a combination between patterns and intents
print(len(documents), 'documents')
#classes = intents
print(len(classes), 'classes', classes)       
#words = all words, vocabulary
print(len(words), 'unique words', words)            


#save words and classes list using pickle
pickle.dump(words,open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#create our taining data
training = []
output_empty =[0] * len(classes)

for doc in documents:
    #initializing bag of words
    bag = []
    #list of tokenized words for the pattern
    pattern_words = doc[0]
    #lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    #create our bag of words array with 1, if word match in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    
    #output is a '0' for each tag and '1' for current tag(for each pattern) 
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
    
#shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

#create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print('Training data created')

#create the deep learning model
#create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output 1
#equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile model Stochastic gradient descent with Nesterov accelarated gradient gives good result for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist= model.fit(np.array(train_x), np.array(train_y), epochs=200,batch_size=5, verbose=1)
model.save('python_bot_model.h5', hist)

print('all done')



