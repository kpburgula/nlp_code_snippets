import re
import unicodedata
import string


class Preprocessing:
    def __init__(self, path):
        self.path = path
        with open(self.path, mode='rt', encoding='utf-8') as data:
            self.raw_data = data.read()[:300000].split('\n')
        self.punctuation_dictionary = str.maketrans('', '', string.punctuation)
        self.regex_filter = re.compile('[^%s]' % re.escape(string.printable))

    def normalization(self, sentences):
        """
        visit https://www.wikiwand.com/en/Unicode_equivalence#/Normal_forms for details about normalization methods
        :param sentence: string
        :return: normalized data
        """
        sents = []
        for sentence in sentences:
            temp = [unicodedata.normalize('NFD', sentence).encode('ascii', 'ignore').decode('UTF-8')]
            sents.append(' '.join(temp))
        return sents

    def remove_punctuation(self, sentences):
        sents = []
        for sentence in sentences:
            temp = [word.translate(self.punctuation_dictionary) for word in sentence.split()]
            sents.append(' '.join(temp))
        return sents

    def remove_noise(self, sentences):
        sents = []
        for sentence in sentences:
            temp = [self.regex_filter.sub('', w) for w in sentence.split()]
            temp = [word for word in temp if word.isalpha()]
            sents.append(' '.join(temp))
        return sents

    # Handle most frequent and in-frequent words
