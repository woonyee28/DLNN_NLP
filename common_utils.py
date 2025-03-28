import json
import os
import random

from datasets import Dataset, load_dataset
import nltk
import torch
import numpy as np


SAVE_DIR = "./result/"
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab.json")
EMBEDDING_PATH = os.path.join(SAVE_DIR, "embedding.json")
EMBEDDING_DIM = 100

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
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
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

    
def load_glove_embeddings() -> dict:
    """
    Load GloVe embeddings
    """
    glove_dict = {}
    word_embedding_glove = load_dataset("SLU-CSCI4750/glove.6B.100d.txt")
    word_embedding_glove = word_embedding_glove["train"]

    for example in word_embedding_glove:
        split_line = example["text"].strip().split()
        word = split_line[0]
        vector = np.array(split_line[1:], dtype="float32")
        glove_dict[word] = vector

    print(f"Total GloVe words loaded: {len(glove_dict)}")
    return glove_dict


def create_embedding_matrix(vocab, save=False) -> dict:
    glove_dict = load_glove_embeddings()
    embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))

    missing_words = []
    for word in vocab:
        if word not in glove_dict:
            missing_words.append(word)
    missing_words_embedding = np.mean(list(glove_dict.values()), axis=0)

    word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    for word, idx in word2idx.items():
        if word in missing_words:
            embedding_matrix[idx] = missing_words_embedding
        else:
            embedding_matrix[idx] = glove_dict[word]
    
    word_to_embedding = {}
    for word, idx in word2idx.items():  
        embedding = embedding_matrix[idx].tolist()
        word_to_embedding[word] = embedding
    
    if save:
        with open(EMBEDDING_PATH, "w", encoding="utf-8") as f:
            json.dump(word_to_embedding, f, ensure_ascii=False, indent=4)

        print(f"Word to Embeddings saved to {EMBEDDING_PATH}")

    return word_to_embedding


