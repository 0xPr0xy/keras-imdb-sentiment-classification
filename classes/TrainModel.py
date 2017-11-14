from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
import json

class TrainModel(object):

    def __init__(self):
        pass

    def create_model(self, num_words, max_sentence_length, embedding_vector_len):
        self.model = Sequential()
        self.model.add(Embedding(num_words, embedding_vector_len, input_length=max_sentence_length))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100))
        self.model.add(Dense(1, activation='sigmoid'))

        return self

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self

    def save_model_as_json(self, path='model.json'):
        model_json = self.model.to_json()
        with open(path, "w") as json_file:
            json_file.write(model_json)

        return self

    def train_model(self, x, y, epochs=3, batch_size=64):
        self.x = x
        self.y = y
        self.model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size)

        return self

    def save_trained_weights(self, path='weights.h5'):
        self.model.save_weights(path)

        return self
