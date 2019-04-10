from misc_utils import track
import numpy as np


class Embedding:
    def __init__(self, path, debug=False):
        self.path = path
        self.debug = debug
        self.embedding_size = None
        self.embedding_mean = None
        self.embedding_std = None
        self.embedding_index = self._load_embeddings()
        self.valid_initializer_values = ["zero", "random"]

    def _load_embeddings(self):
        if self.debug:
            print("Loading embedding file : {0}".format(self.path))
        embedding_index = dict(Embedding._get_coefs(*o.strip().split(" ")) for o in track(open(self.path),
                                                                                          _track=self.debug))
        all_embs = np.stack(embedding_index.values())
        self.embedding_mean, self.embedding_std = all_embs.mean(), all_embs.std()
        self.embedding_size = all_embs.shape[1]
        return embedding_index

    @staticmethod
    def _get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

    def get_embedding_matrix(self, word_index: dict, max_features: int, initializer: str="zero", debug: bool=True):
        """
        Returns numpy 2D array of shape N x D (number of words x number of dimensions)
        :param word_index: words index dict, with word as key and index as it's value
        :param initializer: Embedding matrix initialization criteria.
                    It can have two values : "zero" (initialize by np.zero) or "random" (initialize by np.random.normal)
        :param debug: True or False : to show progress
        :param max_features: Maximum number of words to be considered in embedding matrix

        """
        self.debug = debug
        if self.debug:
            print("Creating embedding matrix for {0} words".format(len(word_index)))

        assert initializer in self.valid_initializer_values, \
            "Valid value for 'initializer' : {0}\nPassed value : {1}".format(self.valid_initializer_values, initializer)

        nb_words = min(max_features, len(word_index))
        if initializer == "zero":
            embedding_matrix = np.zeros((nb_words, self.embedding_size))
        else:
            embedding_matrix = np.random.normal(self.embedding_mean, self.embedding_std, (nb_words, self.embedding_size))
        for word, i in track(word_index.items(), _track=self.debug):
            if i >= max_features:
                continue
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
