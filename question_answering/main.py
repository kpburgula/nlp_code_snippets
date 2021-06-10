from question_answering.utils import *
from question_answering.constants import *
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from tqdm import trange
from torch.utils.data import DataLoader
from transformers import AdamW


# This module is based on DistilBert architecture
class QuestionAnswering:

    def __init__(self, custom=False, model_path='custom_model'):
        if custom:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
        else:
            model_path = 'distilbert-base-uncased'
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
            self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
            self.save_model(model_path)

    def encode(self, contexts, questions, inference=False):
        if inference:
            return self.tokenizer(contexts, questions, truncation=True, padding=True, return_tensors='pt')
        return self.tokenizer(contexts, questions, truncation=True, padding=True)

    def answer_encode(self, encodings, answers):
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = self.tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = self.tokenizer.model_max_length

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

    def train_model(self, dataset):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model.to(device)
        self.model.train()

        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        for _ in trange(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions,
                                     end_positions=end_positions)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
        self.save_model()

    def save_model(self, model_path='custom_model'):
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        self.tokenizer.save_pretrained(model_path)
        self.model.save_pretrained(model_path)

    def prediction(self, question="Who was Jim Henson?", text="Jim Henson was a nice puppet"):
        self.model.eval()
        inputs = self.tokenizer(question, text, return_tensors='pt')
        outputs = self.model(**inputs)
        answer_start = torch.argmax(outputs[0])  # get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(outputs[1]) + 1

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

        return answer

    def batch_prediction(self, inputs, dataset):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Add model to device
        self.model.to(device)
        # switch model out of training mode
        self.model.eval()
        # initialize validation set data loader
        val_loader = DataLoader(dataset, batch_size=prediction_batch)
        answers = []
        # loop through batches
        for batch in val_loader:
            # we don't need to calculate gradients as we're not training
            with torch.no_grad():
                # pull batched items from loader
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # we will use true positions for accuracy calc
                start_true = batch['start_positions'].to(device)
                end_true = batch['end_positions'].to(device)
                # make predictions
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # pull prediction tensors out and argmax to get predicted tokens
                start_pred = torch.argmax(outputs['start_logits'], dim=1)
                end_pred = torch.argmax(outputs['end_logits'], dim=1) - 1
                # calculate the average score using start and end logit scores
                scores = ((torch.max(outputs['end_logits'], dim=1).values + torch.max(outputs['start_logits'],
                                                                                      dim=1).values) / 2)

                for i in range(len(input_ids)):
                    batch_answers = self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(input_ids[i][start_pred[i]:end_pred[i]]))
                    score = scores[i].item()
                    answers.append({
                        'answer': batch_answers,
                        'score': score
                    })
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)
        return answers
