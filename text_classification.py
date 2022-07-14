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

pln = spacy.load('pt')
pln

base_treinamento['tweet_text'][1]

stop_words = spacy.lang.pt.stop_words.STOP_WORDS

print(stop_words)

string.punctuation

def preprocessamento(texto):
  # Letras minúsculas
  texto = texto.lower()

  # Nome do usuário
  texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', texto)

  # URLs
  texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)

  # Espaços em branco
  texto = re.sub(r" +", ' ', texto)

  # Emoticons
  lista_emocoes = {':)': 'emocaopositiva',
                   ':d': 'emocaopositiva',
                   ':(': 'emocaonegativa'}
  for emocao in lista_emocoes:
    texto = texto.replace(emocao, lista_emocoes[emocao])

  # Lematização
  documento = pln(texto)

  lista = []
  for token in documento:
    lista.append(token.lemma_)
  
  # Stop words e pontuações
  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
  
  return lista

texto_teste = '@behin_d_curtain :D Para :( mim, http://www.iaexpert.com.br é precisamente o contrário :) Vem a chuva e vem a boa disposição :)'
resultado = preprocessamento(texto_teste)
resultado

base_treinamento.head(10)

base_treinamento['tweet_text'] = base_treinamento['tweet_text'].apply(preprocessamento)

base_treinamento.head(10)

base_teste['tweet_text'] = base_teste['tweet_text'].apply(preprocessamento)

base_teste.head(10)