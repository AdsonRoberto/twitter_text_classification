!pip install -q spacy==2.2.3

import spacy
spacy.__version__

!python3 -m spacy download pt

import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
import re

base_treinamento = pd.read_csv('/content/Train50.csv', delimiter=';')

from google.colab import drive
drive.mount('/content/drive')

base_treinamento.shape

base_treinamento.head()

base_treinamento.tail()

sns.countplot(base_treinamento['sentiment'], label = 'Contagem');

base_treinamento.drop(['id', 'tweet_date', 'query_used'], axis = 1, inplace=True)

base_treinamento.head()

sns.heatmap(pd.isnull(base_treinamento));

base_teste = pd.read_csv('/content/Test.csv', delimiter=';')

base_teste.head()

base_teste.shape

sns.countplot(base_teste['sentiment'], label='Contagem');

base_teste.drop(['id', 'tweet_date', 'query_used'], axis = 1, inplace=True)

base_teste.head()

sns.heatmap(pd.isnull(base_teste));