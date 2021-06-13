import requests
import tarfile
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
from gensim.models import LdaModel, LsiModel
from gensim.corpora import Dictionary

# Use https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
# to explore more on LDA and LSA

def get_data():
    """
    This function downloads the required data to test lda and lsa models
    """

    url = 'http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz'
    target_path = 'books.tar.gz'
    if not os.path.isfile(target_path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            open(target_path, 'wb').write(response.content)

        # open tar zip file file
        file = tarfile.open(target_path)
        # extracting the tar zip file
        file.extractall('.')
        file.close()
    return True


class TopicModeling:

    def __init__(self, data_path):
        self.data_path = data_path

    def test_pipeline_data(self, doc_list):
        """
        Takes list of strings as input. Applies preprocessing
        :param doc_list: List of strings
        :return: list of lists of tokens
        """
        summaries = []
        for item in doc_list:
            summaries.append(self.preprocess(item))
        return summaries

    # tokenize, remove stopwords, non-alphabetic words, lowercase
    def preprocess(self, text):
        stops = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        return [token.lower() for token in tokens if token.isalpha() and token not in stops]

    def prepare_data(self):
        summaries = []
        for line in open(self.data_path, encoding="utf-8"):
            temp = line.split("\t")
            summaries.append(self.preprocess(temp[6]))

        return summaries

    def get_dictionary(self, summaries):
        # Create a dictionary representation of the documents.
        dictionary = Dictionary(summaries)
        # Filter infrequent or too frequent words.
        dictionary.filter_extremes(no_below=10, no_above=0.5)
        corpus = [dictionary.doc2bow(summary) for summary in summaries]
        temp = dictionary[0]
        return corpus, dictionary

    def train_lda_model(self, corpus, dictionary):
        self.lda_model = LdaModel(corpus=corpus, id2word=dictionary.id2token, iterations=400, num_topics=10)

    def train_lsa_model(self, corpus, dictionary):
        self.lsa_model = LsiModel(corpus=corpus, num_topics=10, id2word=dictionary.id2token)

    def get_top_topics(self, model_name, data):
        """
        Returns the top topics
        :param self: Contains trained models and other required info
        :param model_name: 'lda' or 'lsa'
        :param data: get_dictionary method provides it
        :return: List of top topics
        """

        if model_name == 'lda':
            return list(self.lda_model.top_topics(data))
        elif model_name == 'lsa':
            return list(self.lsa_model.print_topics(num_topics= 10))


if __name__ == "__main__":
    get_data()
    instance = TopicModeling('booksummaries/booksummaries.txt')
    summaries = instance.prepare_data()
    corpus, dict_gensim = instance.get_dictionary(summaries)
    instance.train_lda_model(corpus, dict_gensim)
    instance.train_lsa_model(corpus, dict_gensim)
    print(instance.get_top_topics('lda', corpus))
    print(instance.get_top_topics('lsa', corpus))
