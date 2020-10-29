import re
import os
import sys

from efselabwrapper.tokenizer import build_sentences
from efselabwrapper.stopwords import stopwords
from efselabwrapper.config import get_models

MAX_TOKEN = 256

def run_processing_pipeline(data):
    '''
    INPUT: list of string (corpus)\n
    OUTPUT: list of list of string (documents as lists of of lemmas, filtered on POS and stopwords)
    '''
    
    models = get_models() 
    return [process_document(doc, models) for doc in data]


def run_annotation_pipeline(data):
    '''
    INPUT: list of string (corpus)\n
    OUTPUT: list of list of tuple (documents as lists of tokens)
    '''

    models = get_models()
    return [annotate_document(doc, models) for doc in data]

def annotate_document(doc, models):
    doc_sentences = annotate_sentences(doc, models)
    
    # Flatten document of sentences of tokens into document of tokens, and keep all tokens
    return [token for sentence in doc_sentences for token in sentence]

def process_document(doc, models):
    doc_sentences = annotate_sentences(doc, models)

    # Flatten document of sentences of tokens into document of token.lemma, if token should be kept
    return [clean_word(token[1]) for sentence in doc_sentences for token in sentence if keep_token(token)]


def clean_word(word):
    return word.replace('\xad', '').strip('”').strip('-').replace('-', '_').lower()


def keep_token(token):
    POS = ['PROPN', 'NOUN', 'ADJ']
    #removed ['VERB', 'ADV', 'SCONJ', 'AUX', 'PUNCT', 'ADP', 'PRON','DET', 'PART', 'CCONJ', 'NUM', 'INTJ']
    if token[2] in POS and token[1] not in stopwords and len(token[1]) > 1:
        return True
    else:
        return False


def annotate_sentences(document, models):
    '''
    INPUT:  string (document)\n
    OUTPUT: list of list of tuple (document as a list of sentences of tokens)
    '''
    sentences = run_tokenization(document)
    annotated_sentences = []

    for sentence in sentences:
        lemmas, ud_tags_list, suc_tags_list, suc_ne_list = run_tagging_and_lemmatization(
            sentence, models)
        ud_tag_list = [ud_tags[:ud_tags.find("|")] for ud_tags in ud_tags_list]

        # Token format: (word, lemma, POS, NER-tag)
        annotated_sentences.append([e for e in zip(sentence, lemmas, ud_tag_list, suc_ne_list)])

    return annotated_sentences


def run_tokenization(data, non_capitalized=None):
    sentences = build_sentences(data, non_capitalized=non_capitalized)
    sentences_filtered = list(filter(bool,
                                     [[token for token in sentence if len(token) <= MAX_TOKEN]
                                      for sentence in sentences]))
    return sentences_filtered


def run_tagging_and_lemmatization(sentence, models):
    lemmas = []
    ud_tags_list = []
    suc_tags_list = models["suc_tagger"].tag(sentence)
    suc_ne_list = []

    lemmas = [
        models["lemmatizer"].predict(token, tag)
        for token, tag in zip(sentence, suc_tags_list)
    ]
    ud_tags_list = models["ud_tagger"].tag(sentence, lemmas, suc_tags_list)

    suc_ne_list = models["suc_ne_tagger"].tag(
        list(zip(sentence, lemmas, suc_tags_list))
    )

    return lemmas, ud_tags_list, suc_tags_list, suc_ne_list
