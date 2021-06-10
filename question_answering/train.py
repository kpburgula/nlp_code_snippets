from question_answering.main import *
from question_answering.utils import *
from question_answering.constants import *

if __name__ == "__main__":
    # Initialize QA model
    instance = QuestionAnswering()
    # Get lists of contexts, questions, answers
    train, val = get_data()
    # Encoding
    train_encodings = instance.encode(train[0], train[1])
    val_encodings = instance.encode(val[0], val[1])
    train_encodings = instance.answer_encode(train_encodings, train[2])
    val_encodings = instance.answer_encode(val_encodings, val[2])
    # Convert to pytorch dataset
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    instance.train_model(train_dataset)
