import tensorflow as tf
import pandas as pd
from bert_finetuning_pt.constants import *
from sklearn.model_selection import train_test_split
import numpy as np


def get_gpu_status():
    # Test if GPU is enabled
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('No GPU Found')
        return False
    else:
        return True


def get_data():
    print('Getting training data')
    training_data = pd.read_csv(training_path)

    testing_data = pd.read_csv(testing_path)

    training_data = training_data[['sentence', 'label']]
    testing_data = testing_data[['sentence', 'label']]

    training_data.columns = ['input', 'output']
    testing_data.columns = ['input', 'output']

    return training_data, testing_data


def preprocess_data(df):
    return df


def get_splitted_data(input_ids, attention_masks, labels):
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=43,
                                                                                        test_size=test_size)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=43,
                                                           test_size=test_size)

    return train_inputs, train_labels, validation_inputs, validation_labels, train_masks, validation_masks


def flat_accuracy(preds, labels):
    # Function to calculate the accuracy of our predictions vs labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
