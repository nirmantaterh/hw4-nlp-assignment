import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# QWERTY keyboard layout: each key maps to nearby keys
QWERTY_NEIGHBORS = {
    'q': ['w', 'a'],
    'w': ['q', 'e', 's', 'a'],
    'e': ['w', 'r', 'd', 's'],
    'r': ['e', 't', 'f', 'd'],
    't': ['r', 'y', 'g', 'f'],
    'y': ['t', 'u', 'h', 'g'],
    'u': ['y', 'i', 'j', 'h'],
    'i': ['u', 'o', 'k', 'j'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'a': ['q', 'w', 's', 'z'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'z': ['a', 's', 'x'],
    'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'],
    'v': ['c', 'f', 'g', 'b'],
    'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'],
    'm': ['n', 'j', 'k'],
}


def get_typo_char(char):
    """Replace a character with a nearby QWERTY key."""
    char_lower = char.lower()
    if char_lower not in QWERTY_NEIGHBORS:
        return char
    
    neighbors = QWERTY_NEIGHBORS[char_lower]
    if not neighbors:
        return char
    
    replacement = random.choice(neighbors)
    
    # Preserve case
    if char.isupper():
        replacement = replacement.upper()
    
    return replacement


def tokenize_function(examples):
    """Tokenize function for dataset processing."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Q2: QWERTY Typo Transformation
    # Introduces realistic typos by replacing ~10% of alphabetic characters
    # with nearby QWERTY keys. Sentiment is preserved because:
    # - Only a small fraction of characters are replaced
    # - QWERTY proximity keeps replacements realistic (e.g., "great" → "gerat")
    # - A human would still recognize the sentiment despite typos
    
    text = example["text"]
    
    # Collect indices of alphabetic characters
    alpha_indices = [i for i, c in enumerate(text) if c.isalpha()]
    
    if not alpha_indices:
        return example
    
    # Determine number of characters to corrupt (~10% typo rate)
    typo_rate = 0.10
    num_to_corrupt = max(1, int(len(alpha_indices) * typo_rate))
    indices_to_corrupt = random.sample(alpha_indices, min(num_to_corrupt, len(alpha_indices)))
    
    # Convert to list and apply typos
    text_list = list(text)
    for idx in indices_to_corrupt:
        text_list[idx] = get_typo_char(text_list[idx])
    
    example["text"] = ''.join(text_list)

    ##### YOUR CODE ENDS HERE ######

    return example