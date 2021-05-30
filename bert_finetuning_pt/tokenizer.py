from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from bert_finetuning_pt.constants import *


def get_tokenized_texts(df):
    sentences = df.input.values
    # Adding CLS and SEP tokens at the beginning and end of each sentence for BERT
    sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
    labels = df.output.values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    print("After tokenizing the first sentence:")
    print(tokenized_texts[0])

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    print('After replacing tokens with index numbers in the BERT 30500 vocabulary: ')
    print(input_ids[0])
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    # Use https://huggingface.co/transformers/glossary.html#attention-mask
    # to know better why we use attention masks apart from [SEP] and segment embeddings
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    return input_ids, attention_masks, labels
