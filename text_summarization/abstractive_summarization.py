import requests
import tarfile
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


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

    def __init__(self):
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

    def get_abstractive_summarizer(self, raw_text):
        preprocess_text = raw_text.strip().replace("\n", "")
        tokenized_text = self.tokenizer.encode(preprocess_text, return_tensors="pt")  # pt for pytorch
        summary_ids = self.model.generate(tokenized_text,
                                          num_beams=4,
                                          no_repeat_ngram_size=2,
                                          min_length=30,
                                          max_length=100,
                                          early_stopping=True)
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return output


if __name__ == "__main__":
    get_data()
    text = list(prepare_data("booksummaries/booksummaries.txt").values())[0]
    instance = TextSummarizer()
    print(instance.get_abstractive_summarizer(text))
    print('_' * 50)
