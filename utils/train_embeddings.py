from utils.preprocessing import FetchData, Preprocessing
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import gensim
import os


# Refer https://rare-technologies.com/word2vec-tutorial/ for more details

class TrainEmbeddings(Preprocessing):

    def __init__(self):
        super().__init__()
        self.raw_data = FetchData().get_sample_data()
        self.cleaned_data = []

    def preprocess(self):
        # sentence parsing
        for i in self.raw_data:
            temp = []
            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())
            self.cleaned_data.append(temp)

    def train(self):
        train_model = Word2Vec(self.cleaned_data, min_count=1, size=512, window=5, sg=1)
        # save the model
        if not os.path.isdir('my_model'):
            train_model.save('my_model')


instance = TrainEmbeddings()
instance.preprocess()
instance.train()

# Load the trained model
new_model = gensim.models.Word2Vec.load('my_model')
print(new_model.similarity('Potato', 'Banana'))
