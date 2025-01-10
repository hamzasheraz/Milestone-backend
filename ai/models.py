from django.db import models
import os
import spacy
from transformers import pipeline
import pandas as pd
from sentence_transformers import SentenceTransformer

model_dir = os.path.join(os.getcwd(), 'static','static', 'sentence_transformer')
sentence_model = SentenceTransformer(model_dir)
model_path = os.path.join(os.getcwd(), 'static', 'summarizer')
model_path_classifier = os.path.join(os.getcwd(), 'static', 'zero-shot-classification')
classifier=pipeline("zero-shot-classification", model=model_path_classifier)
summarizer = pipeline("summarization", model=model_path)
nlp=spacy.load(os.path.join(os.getcwd(), 'static', 'spacy_model'))