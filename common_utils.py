import json
import os
import random

from datasets import Dataset, load_dataset, DatasetDict
import nltk
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



SAVE_DIR = "./result/"
VOCAB_PATH = os.path.join(SAVE_DIR, "vocab.json")
EMBEDDING_PATH = os.path.join(SAVE_DIR, "embedding_matrix.npy")
EMBEDDING_DIM = 100
WORD2IDX_PATH = os.path.join(SAVE_DIR, "word2idx.json")
MAX_SEQ_LENGTH = 100

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
    """
    Create embedding matrix for the vocabulary, include PAD and UNK too
    """
    glove_dict = load_glove_embeddings()
    vocab = vocab.union({'<PAD>', '<UNK>'})
    embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))

    missing_words = []
    for word in vocab:
        if word not in glove_dict and word not in {'<PAD>', '<UNK>'}:
            missing_words.append(word)
    missing_words_embedding = np.mean(list(glove_dict.values()), axis=0)

    special_tokens = ['<PAD>', '<UNK>']
    regular_words = sorted([word for word in vocab if word not in special_tokens])
    all_words = special_tokens + regular_words
    word2idx = {word: idx for idx, word in enumerate(all_words)}
    for word, idx in word2idx.items():
        if word == '<PAD>':
            embedding_matrix[idx] = np.zeros(EMBEDDING_DIM)
        elif word == '<UNK>' or word in missing_words:
            embedding_matrix[idx] = missing_words_embedding
        else:
            embedding_matrix[idx] = glove_dict[word]
    
    if save:
        np.save(EMBEDDING_PATH, embedding_matrix)
        print(f"Embedding Matrix saved to {EMBEDDING_PATH}")
        
        with open(WORD2IDX_PATH, "w", encoding="utf-8") as f:
            json.dump(word2idx, f, ensure_ascii=False, indent=4)
        print(f"Word to Index mapping saved to {WORD2IDX_PATH}")

    return embedding_matrix, word2idx


def create_train_validation_test(dataset: Dataset):
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_val = train_test['train'].train_test_split(test_size=0.125, seed=42)  # 0.125 * 0.8 = 0.1

    dataset_dict = {
        'train': train_val['train'],
        'validation': train_val['test'],
        'test': train_test['test']
    }

    print(f"Train size: {len(train_val['train'])}")
    print(f"Validation size: {len(train_val['test'])}")
    print(f"Test size: {len(train_test['test'])}")

    return dataset_dict

def create_dataloaders(dataset_dict, encoded_labels_dict, word2idx, batch_size=32, max_seq_length=100):
    """
    Create PyTorch DataLoaders from the dataset dictionary with pre-encoded labels.
    """
    def text_to_indices(text, max_length=max_seq_length):
        tokens = nltk.word_tokenize(text.lower())
        indices = []
        for token in tokens[:max_length]:
            if token in word2idx:
                indices.append(word2idx[token])
            else:
                indices.append(word2idx['<UNK>'])

        if len(indices) < max_length:
            indices += [word2idx['<PAD>']] * (max_length - len(indices))

        return indices[:max_length]  
    
    dataloaders_dict = {}
    
    for split in ['train', 'validation', 'test']:
        input_ids = torch.tensor([
            text_to_indices(example["text"], max_seq_length) 
            for example in dataset_dict[split]
        ], dtype=torch.long)
        
        labels = torch.tensor(encoded_labels_dict[split], dtype=torch.long)
        
        tensor_dataset = TensorDataset(input_ids, labels)
        
        shuffle = (split == 'train')  # Only shuffle training data
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        dataloaders_dict[split] = dataloader
    
    print(f"Created DataLoaders with {len(dataloaders_dict['train'])} training batches, "
          f"{len(dataloaders_dict['validation'])} validation batches, and "
          f"{len(dataloaders_dict['test'])} test batches.")
    
    return dataloaders_dict