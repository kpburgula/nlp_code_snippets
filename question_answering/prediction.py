from question_answering.main import *
from question_answering.utils import *
from question_answering.constants import *

if __name__ == "__main__":
    # Initialize QA trained model
    instance = QuestionAnswering(custom=True)
    # Get lists of contexts, questions, answers
    train, val = get_data()
    # Encoding
    val_encodings = instance.encode(val[0], val[1], inference=True)
    val_encodings = instance.answer_encode(val_encodings, val[2])
    # Convert to pytorch dataset
    val_dataset = SquadDataset(val_encodings)
    print(instance.batch_prediction(val_encodings, val_dataset))
