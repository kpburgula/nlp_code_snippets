import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification
from constants import *
from utils import *


# Load training data
data = pd.read_csv(training_file_name)

# Split the data
train, test = train_test_split(data, test_size=0.2, random_state= 43)

# Encode the datasets
train_encoded = dataset_prep(train)
test_encoded = dataset_prep(test)

# PreTrained Model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels =2)

# Define model requirements
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer = optimizer, loss = loss, metrics= [metric])

# Train the model
model.fit(train_encoded.shuffle(100).batch(batch_size),
          epochs=1,
          batch_size=batch_size,
          validation_data=test_encoded.shuffle(100).batch(batch_size))

# save the model locally
model.save_pretrained(r"C:\Users\Administrator\Desktop\nlp_code_snippets\bert_finetuning\tmp")

