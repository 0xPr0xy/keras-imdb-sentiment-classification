from keras.datasets import imdb
from keras.models import model_from_json
from TrainModel import TrainModel

class TrainedModel(object):

    def __init__(self):
        pass

    def __get_predictions(self, predictions):
        rounded = [round(x[0]) for x in predictions]
        rounded_indexed = {k: v for k,v in enumerate(rounded)}

        return rounded_indexed

    def __get_input_sentence(self, current_index, id_to_word, test_x, character_to_remove):
        sentence = ' '.join(id_to_word[index] for index in test_x[current_index])
        for character in character_to_remove:
            sentence = sentence.replace(character,'')

        return sentence

    def __get_expected_and_prediction(self, current_index, rounded_indexed, test_y):
        predicted = str(rounded_indexed[current_index])
        expected = str(test_y[current_index])

        return 'predicted: ' + predicted + ' expected: ' + expected

    def retrain(self, dataset):
        train_model = TrainModel()
        (train_model
            .create_model(dataset.num_words, dataset.max_sentence_length, dataset.embedding_vector_len)
            .compile_model()
            .save_model_as_json()
            .train_model(dataset.train_x, dataset.train_y)
            .save_trained_weights()
        )

        return self

    def load_from_json(self, path='model.json'):
        json_file = open(path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        return self

    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self

    def load_weights(self, path='weights.h5'):
        self.model.load_weights(path)

        return self

    def evaluate(self, dataset):
        scores = self.model.evaluate(dataset.train_x, dataset.train_y)
        print('\n%s: %.2f%%' % (self.model.metrics_names[1], scores[1]*100))

        return self

    def predict(self, dataset):
        predictions = self.model.predict(dataset.test_x)
        rounded_indexed = self.__get_predictions(predictions)
        for current_index in rounded_indexed:
                print('\n' + self.__get_expected_and_prediction(current_index, rounded_indexed, dataset.test_y))
                print('sentence: ' + self.__get_input_sentence(current_index, dataset.id_to_word, dataset.test_x, dataset.characters_to_remove))
