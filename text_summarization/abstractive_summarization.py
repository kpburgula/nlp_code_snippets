import requests
import tarfile
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from utils.preprocessing import FetchData


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
    text = FetchData().get_sample_data()[0]
    instance = TextSummarizer()
    print(instance.get_abstractive_summarizer(text))
    print('_' * 50)
