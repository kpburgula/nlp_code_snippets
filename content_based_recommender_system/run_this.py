import requests
import tarfile
import os

from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def get_data():
    """
    This function downloads the required data to test a simple content based recommender system
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


class Recommender:

    def __init__(self, data_path):
        self.data_path = data_path

    def prepare_data(self):
        mydata = {}  # titles-summaries dictionary object
        for line in open(self.data_path, encoding="utf-8"):
            temp = line.split("\t")
            mydata[temp[2]] = temp[6]

        return mydata

    def build_model(self, mydata):
        # prepare the data for doc2vec, build and save a doc2vec model
        train_doc2vec = [TaggedDocument((word_tokenize(mydata[t])), tags=[t]) for t in mydata.keys()]
        model = Doc2Vec(vector_size=50, alpha=0.025, min_count=10, dm=1, epochs=100)
        model.build_vocab(train_doc2vec)
        model.train(train_doc2vec, total_examples=model.corpus_count, epochs=model.epochs)
        model.save("d2v.model")

        return model

    def predict(self, data, model_path="d2v.model"):
        # Use the model to look for similar texts
        model = Doc2Vec.load(model_path)
        new_vector = model.infer_vector(word_tokenize(data))
        similar = model.docvecs.most_similar([new_vector])  # gives 10 most similar titles

        return similar


if __name__ == "__main__":
    get_data()
    instance = Recommender('booksummaries/booksummaries.txt')
    processed_data = instance.prepare_data()
    model = instance.build_model(processed_data)
    preds = instance.predict("This is great movie")
    print(preds)
