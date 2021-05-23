import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer
from constants import *
from transformers import TFBertForSequenceClassification
import numpy as np
from sklearn.metrics import classification_report

# Use the same tokenizer used for training
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def tokenized_outputs(texts):
    return tokenizer(texts,
                    truncation=True,
                    padding=True,
                    add_special_tokens=True,  # add [CLS], [SEP]
                    max_length=max_length,  # max length of the text that can go to BERT
                    pad_to_max_length=True,  # add [PAD] tokens
                    return_attention_mask=True,
                    return_tensors='tf') # tf for tensorflow, pt for pytorch

# Load the fine tuned model
loaded_model = TFBertForSequenceClassification.from_pretrained("/tmp")

# Load the testing data
testing_data = pd.read_csv('test.csv')
sample_texts = testing_data['input'].tolist()

# Tokenize the sentences using the same tokenizer used for training
tokenized = tokenized_outputs(sample_texts)

# Make predictions
test_output = loaded_model.predict(tokenized.data)
tf_prediction = tf.nn.softmax(test_output['logits'], axis=1).numpy()
final_preds = np.argmax(tf_prediction, axis = 1)

print(classification_report(testing_data['output'].tolist(),final_preds))
