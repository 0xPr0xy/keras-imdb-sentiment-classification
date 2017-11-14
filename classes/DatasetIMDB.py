from keras.datasets import imdb
from keras.preprocessing import sequence

class DatasetIMDB(object):

    def __init__(self, num_words=50000, max_sentence_length=100, embedding_vector_len=32):
        """
        initialize and configure the IMDB dataset,
        create training and test datasets and pad the sentences in them,
        create options for reconstructing the review sentence from word indexes
        """
        (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=num_words, index_from=3)

        self.train_x = sequence.pad_sequences(train_x, maxlen=max_sentence_length)
        self.train_y = train_y
        self.test_x = sequence.pad_sequences(test_x, maxlen=max_sentence_length)
        self.test_y = test_y
        self.max_sentence_length = max_sentence_length
        self.num_words = num_words
        self.embedding_vector_len = embedding_vector_len
        self.characters_to_remove = ["<PAD> ", "<START> ", "<UNK> "]

        word_to_id = imdb.get_word_index()
        word_to_id = {key:(value+3) for key,value in  word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2

        self.word_to_id = word_to_id
        self.id_to_word = {value:key for key,value in  word_to_id.items()}

    def pad_sentence(self, sentence):

        return sequence.pad_sequences([sentence], maxlen=self.max_sentence_length)
    
    def get_sentence_for_index(self, data_index):
        """
        get the sentence for a given index which holds a list of word indexes
        """
        sentence = ' '.join(self.id_to_word[index] for index in self.test_x[data_index])
        for character in self.characters_to_remove:
            sentence = sentence.replace(character,'')

        return sentence
