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