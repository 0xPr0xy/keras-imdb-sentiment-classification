from keras.datasets import imdb
from keras.models import model_from_json
from keras.preprocessing import text
from TrainModel import TrainModel

class TrainedModel(object):

    def __init__(self):
        pass

    def __get_predictions(self, predictions):
        """
        get prediction scores rounded to 1 or 0
        """
        rounded = [round(x[0]) for x in predictions]
        rounded_indexed = {k: v for k,v in enumerate(rounded)}

        return rounded_indexed

    def __get_input_sentence(self, index):
        """
        get the sentence the prediction was made on
        """
        return self.dataset.get_sentence_for_index(index)

    def __get_expected_and_prediction(self, current_index, rounded_indexed):
        """
        return a string with the predicted score and the expected score
        """
        predicted = str(rounded_indexed[current_index])
        expected = str(self.dataset.test_y[current_index])

        return 'predicted: ' + predicted + ' expected: ' + expected

    def set_dataset(self, dataset):
        """
        assign a dataset
        """
        self.dataset = dataset

        return self

    def retrain(self):
        """
        recreate, compile, save the model,
        then train and save trained weights
        """
        train_model = TrainModel()
        (train_model
            .create_model(self.dataset.num_words, self.dataset.max_sentence_length, self.dataset.embedding_vector_len)
            .compile_model()
            .save_model_as_json()
            .train_model(self.dataset.train_x, self.dataset.train_y)
            .save_trained_weights()
        )

        return self

    def load_from_json(self, path='model.json'):
        """
        load the model from a json file
        """
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        return self

    def compile(self):
        """
        compile the model
        """
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self

    def load_weights(self, path='weights.h5'):
        """
        load pretrained weights
        """
        self.model.load_weights(path)

        return self

    def evaluate(self):
        """
        evaluate the trained model on the provided dataset
        """
        scores = self.model.evaluate(self.dataset.train_x, self.dataset.train_y)
        print('\n%s: %.2f%%' % (self.model.metrics_names[1], scores[1]*100))

        return self

    def predict(self):
        """
        make predictions on the test dataset after training,
        this will print the expected prediction, the actual prediction,
        and the sentence the prediction was made on
        """
        predictions = self.model.predict(self.dataset.test_x)
        rounded_indexed = self.__get_predictions(predictions)

        for current_index in rounded_indexed:
                print('\n' + self.__get_expected_and_prediction(current_index, rounded_indexed))
                print('sentence: ' + self.__get_input_sentence(current_index))

    def one_hot(self, text_input):
        """
        make prediction on custom sentence
        """
        indexed_sentence = text.one_hot(text_input, self.dataset.num_words)
        padded_indexed_sentence = self.dataset.pad_sentence(indexed_sentence)
        predictions = self.model.predict(padded_indexed_sentence)
        print(predictions)
        rounded_indexed = self.__get_predictions(predictions)
        print('prediction: '+str(rounded_indexed[0]))
        print('sentence: '+text_input)

        return self
