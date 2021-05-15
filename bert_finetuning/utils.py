from transformers import BertTokenizer
import tensorflow as tf
from constants import *

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def tokenized_outputs(texts):
    return tokenizer(texts,
                    truncation=True,
                    padding=True,
                    add_special_tokens=True,  # add [CLS], [SEP]
                    max_length=max_length,  # max length of the text that can go to BERT
                    pad_to_max_length=True,  # add [PAD] tokens
                    return_attention_mask=True)

def dataset_prep(df):
    input_texts = df['input'].tolist()
    output_texts = df['output'].tolist()

    encodings = tokenized_outputs(input_texts)
    return tf.data.Dataset.from_tensor_slices((dict(encodings),output_texts))
