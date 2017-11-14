import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import json

numpy.random.seed(7)

# DATASET
# load top 5000 words and split them 50 - 50 into training and evaluation dataset
num_words=50000
index_from=3 # default, but make explicit since we use it later on
(train_x, train_y), (eval_x, eval_y) = imdb.load_data(num_words=num_words, index_from=index_from)

# truncate and pad the train and eval input data to 500 character sentences
# this is needed so the output vectors are of the same size
max_len=100
eval_x = sequence.pad_sequences(eval_x, maxlen=max_len)
train_x = sequence.pad_sequences(train_x, maxlen=max_len)

# LOAD MODEL
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# MODEL
# embedding_vector_len=32
# model = Sequential()
# model.add(Embedding(num_words, embedding_vector_len, input_length=max_len))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))

# COMPILE MODEL
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# SAVE MODEL
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# MODEL TRAINING
# model.fit(train_x, train_y, epochs=3, batch_size=64)

# SAVE WEIGHTS
# model.save_weights("model.h5")

# LOAD WEIGHTS
model.load_weights("model.h5")

# PREDICTION
scores = model.evaluate(train_x, train_y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(eval_x)
rounded = [round(x[0]) for x in predictions]
rounded_indexed = {k: v for k,v in enumerate(rounded)}

word_to_id = imdb.get_word_index()
word_to_id = {k:(v+index_from) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}

for item in rounded_indexed:
    print('\npredicted: '+str(rounded_indexed[item])+' actual: '+str(eval_y[item]))
    sentence = ' '.join(id_to_word[id] for id in eval_x[item])
    sentence = sentence.replace('<PAD> ','')
    sentence = sentence.replace('<START> ', '')
    sentence = sentence.replace('<UNK> ', '?')
    print('sentence: '+sentence)
