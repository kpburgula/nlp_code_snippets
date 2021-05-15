import pandas as pd
import tensorflow as tf
from constants import *
from sklearn.model_selection import train_test_split
from encoding import *

# Load training data
data = pd.read_csv(file_name)

train, test = train_test_split(data, test_size=0.2, random_state= 43)

train_encoded = encode_examples(train)
test_encoded = encode_examples(test)