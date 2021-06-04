from machine_translation.utils import Preprocessing
from constants import *
import pickle


def save_data(path, filename):
    preprocess = Preprocessing(path)
    print('Raw data: \n', preprocess.raw_data[:100])
    clean_data = preprocess.normalization(preprocess.raw_data)
    clean_data = preprocess.remove_punctuation(clean_data)
    clean_data = preprocess.remove_noise(clean_data)
    print('Clean data: \n', clean_data[:100])
    with open(f'{root}/{filename}.pkl', 'wb') as file:
        pickle.dump(clean_data, file)
    print('*' * 100)


save_data(fr_path, 'French')
save_data(en_path, 'English')

