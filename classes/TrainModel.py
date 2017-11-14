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
        """
        create a neural net for classifying imdb reviews with a score in range (0,1)
        """
        self.model = Sequential()
        self.model.add(Embedding(num_words, embedding_vector_len, input_length=max_sentence_length))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100))
        self.model.add(Dense(1, activation='sigmoid'))

        return self

    def compile_model(self):
        """
        compile model with default optimization and accurracy
        """
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self

    def save_model_as_json(self, path='model.json'):
        """
        write the created model to json
        """
        json = self.model.to_json()
        with open(path, "w") as outfile:
            outfile.write(json)

        return self

    def train_model(self, x, y, epochs=3, batch_size=64):
        """
        train the model for a number of epochs
        (3 should give enough accuracy)
        """
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)

        return self

    def save_trained_weights(self, path='weights.h5'):
        """
        save the trained weights
        so the model does not have to be retrained every run
        """
        self.model.save_weights(path)

        return self
