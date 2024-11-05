"""
Utility functions for HW 6

Developer: Katie Keith
"""

import typing
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset

def load_attention_data(toks):
	"""
	Loads the matrices that have already been saved 
	""" 
	X = np.load('data/X.npy')
	W_Q = np.load('data/W_Q.npy')
	W_K = np.load('data/W_K.npy')
	W_V = np.load('data/W_V.npy')

	assert len(toks) == X.shape[0]
	assert X.shape[1] == W_Q.shape[0] == W_V.shape[0] == W_K.shape[0]

	return X, W_Q, W_K, W_V

def substitute_winograde_data(dataset):
    """
    In this preprocessing function for classification, we create two training 
    examples per original example by substituting option1 or option2 for _. 
    
    For instance, with the example 
        {'sentence': 'I had to read an entire story for class tomorrow. 
                        Luckily, the _ was short.', 
        'option1': 'story', 'option2': 'class', 'answer': '1'}
        
    We create 
        x1 = "I had to read an entire story for class tomorrow. Luckily, the story was short."
        y1= 1 (correct)
        
        x2 = "I had to read an entire story for class tomorrow. Luckily, the class was short."
        y2 = 0 (incorrect)
    """
    
    out = []
    for i, x in enumerate(dataset):
        new_x = {}
        # Create a "positive answer"
        if x['answer'] == '1': 
            replace_str = x['option1']
        else: 
            replace_str = x['option2']
        new_x['text'] = x['sentence'].replace("_", replace_str)
        new_x['label'] = 1 # y=1
        out.append(new_x)

        #Then a "negative answer"
        if x['answer'] == '1': 
            replace_str = x['option2'] # Opposite!!! 
        else: 
            replace_str = x['option1']
        new_x['text'] = x['sentence'].replace("_", replace_str)
        new_x['label'] = 0 # y=0
        out.append(new_x)
    return out