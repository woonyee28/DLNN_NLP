import json
import os
import random

from datasets import Dataset
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

SAVE_DIR = "./result/"
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab.json")
EMBEDDINGS_PATH = os.path.join(SAVE_DIR, "embeddings.json")

def tokenize(dataset: Dataset, save=False) -> set:
    """
    Tokenize the text into tokens using Bert Tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    vocab = set()

    for example in dataset:
        tokens = tokenizer(example["text"], return_tensors='pt')
        vocab.update(tokens)

    print(f"Vocabulary size: {len(vocab)}")

    if save:
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(list(vocab), f, ensure_ascii=False, indent=4)

        print(f"Vocabulary saved to {VOCAB_PATH}")
    return vocab

def create_embeddings(tokens: set, save=False) -> list:
    """
    Convert the tokens to embeddings using Bert Model
    """
    model = BertModel.from_pretrained("bert-base-cased")
    embeddings = []

    for encoded_input in tokens:
        output_embeddings = model(encoded_input)
        embeddings.append(output_embeddings)
    
    if save:
        with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=4)
        
        print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    return embeddings


def set_seed(seed=0):
    """
    set random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

