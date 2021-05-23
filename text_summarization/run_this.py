import requests
import tarfile
import os
from summa import summarizer
from summa import keywords
from summarizer import Summarizer
from summarizer.coreference_handler import CoreferenceHandler


def get_data():
    """
    This function downloads the required data to test a text summarization technique
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


def prepare_data(data_path):
    mydata = {}
    for line in open(data_path, encoding="utf-8"):
        temp = line.split("\t")
        mydata[temp[2]] = temp[6]

    return mydata


class TextSummarizer:

    def __init__(self, text):
        self.text = text

    def get_summa_summarizer(self):
        return self.text, summarizer.summarize(self.text, ratio=0.1)

    def get_extractive_summarizer(self):
        model = Summarizer()
        result_1 = model(self.text, min_length=200, ratio=0.01)
        handler = CoreferenceHandler(greedyness=.35)
        model = Summarizer(sentence_handler=handler)
        result_2 = model(self.text, min_length=200, ratio=0.01)

        return self.text, ''.join(result_1), ''.join(result_2)


if __name__ == "__main__":
    get_data()
    text = list(prepare_data("booksummaries/booksummaries.txt").values())[0]
    instance = TextSummarizer(text)
    print(instance.get_summa_summarizer())
    print(instance.get_extractive_summarizer())
    print('_' * 50)
