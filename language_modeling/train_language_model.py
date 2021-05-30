from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, \
                        LineByLineTextDataset, DataCollatorForLanguageModeling,\
                        Trainer, TrainingArguments
from language_modeling.constants import *

# Define the configuration of the Model - BertConfig, RobertaConfig, etc.
# Experiment this part. The config shown below is same as DistilbertConfig
config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Load the Tokenizer in Transformers
tokenizer = RobertaTokenizer.from_pretrained("./CBERT", max_length=512)

# Load the Model From Scratch
model = RobertaForMaskedLM(config=config)
print("Number of parameters to be trained: ", model.num_parameters())

# Build the dataset
dataset = LineByLineTextDataset(
                                tokenizer=tokenizer,
                                file_path=f"{path}/news.txt",
                                block_size=128,)

# Data collator will take samples from the dataset
# and collate them into batches
data_collator = DataCollatorForLanguageModeling(
                                                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
                                            )
# Initializing the Trainer
training_args = TrainingArguments(
    output_dir="./CBERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Pre-train the Model
trainer.train()

# Save the model
trainer.save_model("./CBERT")

# Prediction pipeline
from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./CBERT",
    tokenizer="./CBERT"
)

print(fill_mask("This is a big <mask>."))
