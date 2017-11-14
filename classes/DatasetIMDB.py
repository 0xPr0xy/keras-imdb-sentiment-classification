from keras.datasets import imdb
from keras.preprocessing import sequence

class DatasetIMDB(object):

    def __init__(self):
        pass

    def configure(self, num_words=50000, max_sentence_length=100, embedding_vector_len=32):
        (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=num_words, index_from=3)
        self.train_x = sequence.pad_sequences(train_x, maxlen=max_sentence_length)
        self.train_y = train_y
        self.test_x = sequence.pad_sequences(test_x, maxlen=max_sentence_length)
        self.test_y = test_y
        self.max_sentence_length = max_sentence_length
        self.num_words = num_words
        self.embedding_vector_len = embedding_vector_len

        word_to_id = imdb.get_word_index()
        word_to_id = {key:(value+3) for key,value in  word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2

        self.character_to_remove = ["<PAD> ", "<START> ", "<UNK> "]
        
        id_to_word = {value:key for key,value in  word_to_id.items()}

        self.word_to_id = word_to_id
        self.id_to_word = id_to_word

        return self
