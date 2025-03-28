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
EMBEDDING_BATCH_SIZE = 16
MAX_LENGTH = 512

def tokenize(dataset: Dataset, save=False) -> list:
    """
    Tokenize the text into tokens using Bert Tokenizer
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    vocab = set()
    all_tensors = []

    for example in dataset:
        encoded = tokenizer(
            example["text"], 
            return_tensors='pt',
            padding='max_length',
            max_length=MAX_LENGTH,
            truncation=True
        )
        all_tensors.append({
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        })

        token_ids = encoded['input_ids'][0].tolist()
        vocab.update(token_ids)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of tokenized examples: {len(all_tensors)}")

    if save:
        os.makedirs(os.path.dirname(VOCAB_PATH), exist_ok=True)

        vocab_dict = {
            token_id: tokenizer.convert_ids_to_tokens([token_id])[0] 
            for token_id in vocab
        }

        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=4)

        print(f"Vocabulary saved to {VOCAB_PATH}")
    return all_tensors

def create_embeddings(tokens: list, save=False) -> list:
    """
    Convert the tokens to embeddings using Bert Model
    """
    model = BertModel.from_pretrained("bert-base-cased")
    embeddings = []
    model.eval()
    total_tokens = len(tokens)

    with torch.no_grad():
        for i in range(0, total_tokens, EMBEDDING_BATCH_SIZE):
            batch_end = min(total_tokens, i + EMBEDDING_BATCH_SIZE)
            current_batch = tokens[i:batch_end]
            print(f"Processing batch {i//EMBEDDING_BATCH_SIZE + 1}/{(total_tokens + EMBEDDING_BATCH_SIZE - 1)//EMBEDDING_BATCH_SIZE} "
                  f"(tokens {i+1}-{batch_end}/{total_tokens})")
            batch_input_ids = torch.cat([item['input_ids'] for item in current_batch], dim=0)
            batch_attention_mask = torch.cat([item['attention_mask'] for item in current_batch], dim=0)
            
            batch_outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)

            for j in range(len(current_batch)):
                example_output = batch_outputs[j:j+1]
                embeddings.append(example_output)
    
    if save:
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
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
    

