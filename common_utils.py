import json
import os
import random

from datasets import Dataset
import nltk
import torch
import numpy as np

SAVE_DIR = "./result/"
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab.json")

def tokenize(dataset: Dataset, save=False) -> set:
    """
    Tokenize the text into tokens using nltk
    """
    vocab = set()

    for example in dataset:
        tokens = nltk.word_tokenize(example["text"])
        vocab.update(tokens)

    print(f"Vocabulary size: {len(vocab)}")

    if save:
        with open(VOCAB_PATH, "w", encoding="etf-8") as f:
            json.dump(list(vocab), f, ensure_ascii=False, indent=4)

        print(f"Vocabulary saved to {VOCAB_PATH}")
    return vocab


def set_seed(seed=0):
    """
    set random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    

