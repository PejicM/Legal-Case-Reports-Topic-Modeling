# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:44:43 2020

@author: Marko Pejić
"""


import os
import re
import matplotlib.pyplot as plt
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Phrases, TfidfModel, CoherenceModel
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.wrappers import LdaMallet
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import en_core_web_sm
from pprint import pprint
#nltk.download('stopwords')
#nlp = en_core_web_sm.load()
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nltk.download('wordnet')
stop_words = stopwords.words('english')

# env varijabla za mallet (mogu je kasnije obrisati)
os.environ['MALLET_HOME'] = 'C:\\mallet\\mallet-2.0.8'


# Ucitavanje podataka
def load_data(folder_path):
    data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = folder_path + '/' + filename
            
            with open(file_path, 'r') as f:
                text = f.read()
            f.closed
            
            text = text.strip().replace('\n', '')
            sentences = re.findall(r'<sentence .*?>(.*?)</sentence>+', text)
            name = re.findall(r'<name>(.*?)</name>+', text)[0]

            # ako bude trebao name vrati ovo
            # U TOM SLUCAJU procitaj komentar ispod
            # documents je lista dokumenata, gde je svaki dokument lista od dva elementa: prvi je naslov, a drugi lista recenica!
            #data.append([name, sentences])
            
            data.append(sentences)
            
    return data

# pretvara listu recenica svakog od dokumenata u plaintext
def create_plaintext_from_sentences(documents):
    data = []
    
    for document in documents:
        data.append(' '.join(document))
        
    return data

# Pretprocesiranje
#def lemmatize_stemming(text, stemmer=PorterStemmer(), lemmatizer=WordNetLemmatizer()):
#    return stemmer.stem(lemmatizer.lemmatize(text, pos='v'))       # lematizacija glagola i onda stemming svega

# tokenization and stopwords removing 
def preprocess(text):
    result = []
    #stemmer = PorterStemmer()
    #lemmatizer = WordNetLemmatizer()
    
    for token in simple_preprocess(text):                       # deacc=True za uklanjanje znakova interpunkcije
        if token not in stop_words and len(token) > 3:
            result.append(token)
            
    return result

def lemmatization(documents, allowed_pos_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    documents_out = []
    
    for document in documents:
        if(len(document) > 100000 or len(document) < 10):      # da ne obradjuje prevelike dokumente, ima problema sa memorijom, a i svakako mi ne trebaju
            print(len(document))
        else:
            doc = nlp(' '.join(document))
            documents_out.append([token.lemma_ for token in doc if token.pos_ in allowed_pos_tags])
        
    return documents_out

# Creating bigram and trigram models
def create_ngram_models(documents):
    bigram = Phrases(documents, min_count=5, threshold=100)
    trigram = Phrases(bigram[documents], threshold=100)
    
    bigram_model = Phraser(bigram)
    trigram_model = Phraser(trigram)
    
    return bigram_model, trigram_model

def make_bigrams(documents, bigram_model):
    return [bigram_model[document] for document in documents]

def make_trigrams(documents, trigram_model):
    return [trigram_model[document] for document in documents]

def create_dictionary(documents):
    return Dictionary(documents)

def create_corpus(documents, dictionary):
    return [dictionary.doc2bow(document) for document in documents]

def build_lda_model(dictionary, corpus):
     lda_model = LdaModel(corpus=corpus,
                          id2word=dictionary,
                          num_topics=20,
                          random_state=33,
                          update_every=1,
                          chunksize=100,
                          passes=10,
                          alpha=0.1,
                          eta=0.001,
                          per_word_topics=True)
     return lda_model
 
def find_best_number_of_topics(dictionary, corpus, texts, limit, start=2, step=3):    
    coherence_values = []
    model_list = []
    i = 0
    
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=num_topics,
                         random_state=33,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
        i += 1
        print('Done ' + str(i))
        
    return model_list, coherence_values

def plot_coherence_scores(coherence_values, limit, start=2, step=3):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')


####### Test
documents = load_data('corpus/fulltext')
print('\n'.join(documents[0]))                                              # ispis svih recenica prvog dokumenta
    
documents = create_plaintext_from_sentences(documents)
print(documents[0])

# traje neko vreme
preprocessed_docs = []
for document in documents:
    preprocessed_docs.append(preprocess(document))
    
print(' '.join(preprocessed_docs[0]))

bigram_model, trigram_model = create_ngram_models(preprocessed_docs)
print(trigram_model[bigram_model[preprocessed_docs[0]]])

bigrams = make_bigrams(preprocessed_docs, bigram_model)
trigrams = make_trigrams(preprocessed_docs, trigram_model)      # za sad ih ne koristim, radim sa bigramima

# traje dosta
data_lemmatized = lemmatization(bigrams, allowed_pos_tags=['NOUN', 'ADJ', 'VERB', 'ADV'])
texts = data_lemmatized       # VAZNO!!!

dictionary = create_dictionary(data_lemmatized)
corpus = create_corpus(data_lemmatized, dictionary)

# Standard LDA gensim
lda_model = build_lda_model(dictionary, corpus)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# evaluation
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))       # sto nize to bolje

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Find best number of topics
model_list, coherence_values = find_best_number_of_topics(dictionary=dictionary, corpus=corpus, texts=texts, start=2, limit=40, step=6)
plot_coherence_scores(coherence_values, 40, 2, 6)


# LDA Mallet model
mallet_path = 'C:/mallet/mallet-2.0.8/bin/mallet.bat'
lda_mallet = LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=dictionary, alpha=0.1)

print('\nPerplexity: ', lda_model.log_perplexity(corpus))       # sto nize to bolje
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


