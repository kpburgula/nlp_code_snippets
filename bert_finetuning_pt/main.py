import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel, BertConfig
from bert_finetuning_pt.utils import *
from bert_finetuning_pt.constants import *
from bert_finetuning_pt.tokenizer import *

# specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if get_gpu_status():
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

# Get training data
training_data, testing_data = get_data()
print(f'Training data: {training_data.shape}\n', training_data['output'].value_counts())
print(f'Testing data: {testing_data.shape}\n', testing_data['output'].value_counts())

# Preprocess training data
training_data = preprocess_data(training_data)

# Get tokenized text (Word piece tokenizer for BERT), attention masks, and labels
input_ids, attention_masks, labels = get_tokenized_texts(training_data)

# Get splitted data (Training and validation)
train_inputs, train_labels, validation_inputs, validation_labels, train_masks, validation_masks = get_splitted_data(
    input_ids, attention_masks, labels)

# convert to torch tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Create an iterator of our data with torch DataLoader.
# This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

# Load the bert model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
if get_gpu_status():
    model.cuda()  # model and inputs should be on same device
# Define the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)

# Calculate the total training steps
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
# Refer transformers.get_linear_schedule_with_warmup (documentation) to see the graph
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)


# Training pipeline
def train(model, train_dataloader, validation_dataloader):
    list_losses = []
    epoch_loss = 0
    model.train()
    for batch in train_dataloader:
        # add batch to configured device
        batch = tuple(t.to(device) for t in batch)

        # unpack the inputs from the dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Remove stored gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # Get loss from outputs
        loss = outputs['loss']

        # Backward pass
        loss.backward()

        # Update the parameters and take a step using the computed gradient and learning rate
        optimizer.step()
        scheduler.step()

        # Before moving lets update the batch loss
        epoch_loss = epoch_loss + loss.item()

        # Store the loss of each batch for plots
        list_losses.append(loss.item())

    epoch_loss = epoch_loss / len(train_dataloader)
    val_loss, val_accuracy = predictor(model, validation_dataloader)

    return model, list_losses, {'training_loss': epoch_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy}


def predictor(model, dataloader):
    # Evaluate the model after this epoch
    model.eval()

    val_loss, val_accuracy = 0, 0
    num_steps = 0

    for batch in dataloader:
        # add batch to configured device
        batch = tuple(t.to(device) for t in batch)

        # unpack the inputs from the dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Gradients are not computed
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

        # Move logits and labels to CPU
        logits = outputs['logits'].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # calculate the accuracy and increment a step
        val_accuracy_batch = flat_accuracy(logits, label_ids)
        val_accuracy = val_accuracy + val_accuracy_batch
        val_loss = val_loss + outputs['loss'].item()
        num_steps = num_steps + 1

    val_accuracy = val_accuracy / num_steps
    val_loss = val_loss / len(dataloader)

    return val_loss, val_accuracy


# Run the model for certain number of epochs
track_loss = []
for _ in trange(epochs):
    model, model_history, model_tracker = train(model, train_dataloader, validation_dataloader)
    print(model_tracker)
    track_loss = track_loss + model_history

plt.plot(track_loss)
plt.ylabel('Training loss')
plt.show()
# Prediction pipeline
# Preprocess training data
testing_data = preprocess_data(testing_data)

# Get tokenized text (Word piece tokenizer for BERT), attention masks, and labels
input_ids, attention_masks, labels = get_tokenized_texts(testing_data)

# Convert to torch tensors
test_inputs = torch.tensor(input_ids)
test_labels = torch.tensor(labels)
test_masks = torch.tensor(attention_masks)

# Dataloader for testset
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
test_loss, test_accuracy = predictor(model, test_dataloader)
print(f'On test set, Loss is {test_loss} and Accuracy is {test_accuracy}')
