import json
from pathlib import Path
from question_answering.constants import *
import os
import torch


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def read_squad(path):
    """
    Prepares the squad dataset in certain format
    :param path:
    :return:
    """
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts[:samples], questions[:samples], answers[:samples]


def add_end_idx(answers, contexts):
    """
    Adjust the start and end indexes
    :param answers:
    :param contexts:
    :return:
    """
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx - 1:end_idx - 1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2:end_idx - 2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2  # When the gold label is off by two characters


def get_data():
    train_contexts, train_questions, train_answers = read_squad(f'{root}/train-v2.0.json')
    val_contexts, val_questions, val_answers = read_squad(f'{root}/dev-v2.0.json')
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)
    return [train_contexts, train_questions, train_answers], [val_contexts, val_questions, val_answers]
