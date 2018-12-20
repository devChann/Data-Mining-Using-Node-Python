import spacy
from spacy.lang.en import English
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim import corpora
import pickle
import gensim

spacy.load('en')

parser = English()
text_data = [['jays', 'place', 'great', 'subway'],
             ['described', 'communicative', 'friendly'],
             ['great', 'sarahs', 'apartament', 'back'],
             ['great', 'sarahs', 'apartament', 'back']]


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


nltk.download('wordnet')

''' get the various of the words like synonyms '''
"""
Constructing – (Lemmatization) -> Construct
Extracts – (Lemmatization) -> Extract
Jumping – (Lemmatization) -> Jump
"""


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


''' get the root word'''


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


''' remove stopwords'''
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

''' prepare lDA model for topic modelling'''


def lida_model(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token > 4)]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
print(dictionary)

pickle.dump(corpus, open('corpus.kl', 'wb'))
dictionary.save('dictionary.gensim')

''' find two topics from the text above'''
''' Latent Dirichlet Allocation'''
number_of_topics = 5

idamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=number_of_topics, id2word=dictionary, passes=15)
idamodel.save('idamodel5')
topics = idamodel.print_topic(topicno=4)
for topic in topics:
    print(topic)
