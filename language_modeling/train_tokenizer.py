from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import glob
from language_modeling.constants import *

# load the list of paths where the data is present
paths = list(glob.glob(f'{path}/*.txt'))

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# save model
tokenizer.save_model('.', 'CBERT')

# Test the tokenizer
# Load the tokenizer which is trained on the new texts
tokenizer = ByteLevelBPETokenizer(
    "CBERT/vocab.json",
    "CBERT/merges.txt",
)

# postprocessing step
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("My name is KPBurgula.").tokens)
