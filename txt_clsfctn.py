# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:23:36 2019

@author: Sharmin
"""

# libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
np.random.seed(32)
import nltk


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
from keras.utils.np_utils import to_categorical


MAX_NB_WORDS = 20000

# get the raw text data
##texts_train = train_text.astype(str)
##texts_test = test_text.astype(str)
from sklearn.datasets import load_files

reviews = load_files('txt_sentoken/')
X,y = reviews.data,reviews.target
for i in range(0,len(X)):
        words = nltk.word_tokenize(X[i].decode('utf-8'))
        X[i]=' '.join(words)

texts_train, texts_test, train_y, test_y = train_test_split(X, y, test_size = 0.20, random_state = 0)
# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, char_level=False)
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

index_to_word = dict((i, w) for w, i in tokenizer.word_index.items())

" ".join([index_to_word[i] for i in sequences[0]])


seq_lens = [len(s) for s in sequences]
print("average length: %0.1f" % np.mean(seq_lens))
print("max length: %d" % max(seq_lens))

#matplotlib inline
import matplotlib.pyplot as plt

plt.hist(seq_lens, bins=50);

plt.hist([l for l in seq_lens if l < 200], bins=50);

MAX_SEQUENCE_LENGTH = 150

# pad sequences with 0s
x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x_train.shape)
print('Shape of data test tensor:', x_test.shape)


y_train = train_y
y_test = test_y

y_train = to_categorical(np.asarray(y_train))
print('Shape of label tensor:', y_train.shape)

from keras.layers import Dense, Input, Flatten
from keras.layers import GlobalAveragePooling1D, Embedding
from keras.models import Model

EMBEDDING_DIM = 50
N_CLASSES = 5

# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(MAX_NB_WORDS, EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
embedded_sequences = embedding_layer(sequence_input)

average = GlobalAveragePooling1D()(embedded_sequences)
predictions = Dense(N_CLASSES, activation='softmax')(average)

model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=10, batch_size=128)

output_test = model.predict(x_test)
print("test auc:", roc_auc_score(y_test,output_test[:,1]))

###################################################
avroutput_test=[]
for i in range(len(y_test)):
    if output_test[i][0]>output_test[i][1]:
        avroutput_test.append(0)
    else:
        avroutput_test.append(1)
print(output_test)
print(y_test)
print(len(output_test))



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, avroutput_test)
acc=(cm[0][0]+cm[1][1])/400
##########################################################




# input: a sequence of MAX_SEQUENCE_LENGTH integers
#A complex model : LSTM
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
predictions = Dense(5, activation='softmax')(x)


model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=2, batch_size=128)

output_test = model.predict(x_test)
#########print("test auc:", roc_auc_score(y_test,output_test[:,1]))

##A more complex model : CNN - LSTM
# input: a sequence of MAX_SEQUENCE_LENGTH integers
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# 1D convolution with 64 output channels
x = Conv1D(64, 5)(embedded_sequences)
# MaxPool divides the length of the sequence by 5
x = MaxPooling1D(5)(x)
x = Dropout(0.2)(x)
x = Conv1D(64, 5)(x)
x = MaxPooling1D(5)(x)
# LSTM layer with a hidden size of 64
x = Dropout(0.2)(x)
x = LSTM(64)(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(sequence_input, predictions)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, validation_split=0.1,
          nb_epoch=5, batch_size=128)

output_test = model.predict(x_test)
#print("test auc:", roc_auc_score(y_test,output_test[:,1]))
from sklearn.metrics import confusion_matrix
nbcm = confusion_matrix(y_test, output_test)


#Visualize the outputs of our own Embeddings
from keras import backend as K
get_emb_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].input])
embedding_output = get_emb_layer_output([x_test[:3000]])[0]


emb_shape = embedding_output.shape
to_plot_embedding = embedding_output.reshape(emb_shape[0],emb_shape[1]*emb_shape[2])
y = y_test[:3000]



sentence_emb_tsne = TSNE(perplexity=30).fit_transform(to_plot_embedding)
print(sentence_emb_tsne.shape)
print(y.shape)



plt.figure()
plt.scatter(sentence_emb_tsne[np.where(y == 0), 0],
                   sentence_emb_tsne[np.where(y == 0), 1],
                   marker='x', color='g',
                   linewidth='1', alpha=0.8, label='Happy')
plt.scatter(sentence_emb_tsne[np.where(y == 1), 0],
                   sentence_emb_tsne[np.where(y == 1), 1],
                   marker='v', color='r',
                   linewidth='1', alpha=0.8, label='Unhappy')

plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.title('T-SNE')
plt.legend(loc='best')
plt.savefig('1.png')
plt.show() 










