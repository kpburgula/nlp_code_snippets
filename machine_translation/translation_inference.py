import os
import numpy as np
import trax

# Load the pretrained translation model from Trax
# Each encoder stack and decoder stack contains 6 layers and 8 heads
model = trax.models.Transformer(
 input_vocab_size=33300,d_model=512, d_ff=2048,
 n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
 max_len=2048, mode='predict')

# Initialize the model with pre-trained weights

# Tokenize the sentences

# Decoding the sentences

# De-tokenize the translation and return translated sentence
