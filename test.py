import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

labels = ['good', 'bad']
tokenizer = Tokenizer()
for label in labels:
    print(tokenizer.encode(label, bos=False, eos=False))