from tokenizer import build_sentences
import re
import os
import sys

import lemmatize
from tagger import SucTagger, SucNETagger, UDTagger


DUMMY_STRING = 'Barnen ska gå till skolan, även om regnet faller. Så ska det vara enligt Johan Blad. Men det är på ljug.'
LONG_STRING = '''
Till en början stack smittspridningen ut bland 20–29-åringar, för att sedan nå alla åldersgrupper.

– Smittspridningen gäller inte bara specifika situationer, miljöer eller arbetsplatser. Här är det mer en allmänspridning i samhället, säger hon.

De regioner som har högst smittspridning per capita är Uppsala, Örebro, Jämtland-Härjedalen, Östergötland, Jönköping och Stockholm, enligt Folkhälsomyndighetens senaste lägesrapport. Även Skåne har sett en stor ökning av antalet fall
'''

MAX_TOKEN = 256

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

#PARSING_MODEL = os.path.join(MODEL_DIR, "old-swe-ud")
#MALT = os.path.join(MODEL_DIR, "maltparser-1.9.0/maltparser-1.9.0.jar")

ANNOTATION = 'annotation'
PROCESS = 'process'


def main():
    print("-- MAIN --")
    corpus = [DUMMY_STRING, LONG_STRING]
    processed = run_pipeline(corpus)
    print("DONE! output:")
    print(processed)


def get_models():
    # Place in config.py file
    MODEL_DIR = os.path.join(SCRIPT_DIR, "bin")
    TAGGING_MODEL = os.path.join(MODEL_DIR, "suc.bin")
    NER_MODEL = os.path.join(MODEL_DIR, "suc-ne.bin")
    UD_TAGGING_MODEL = os.path.join(MODEL_DIR, "suc-ud.bin")
    LEMMATIZATION_MODEL = os.path.join(MODEL_DIR, "suc-saldo.lemmas")

    models = {
        "suc_ne_tagger": None,
        "suc_tagger": None,
        "ud_tagger": None,
        "lemmatizer": None,
    }
    models["suc_tagger"] = SucTagger(TAGGING_MODEL)
    models["suc_ne_tagger"] = SucNETagger(NER_MODEL)
    models["ud_tagger"] = UDTagger(UD_TAGGING_MODEL)
    models["lemmatizer"] = lemmatize.SUCLemmatizer()
    models["lemmatizer"].load(LEMMATIZATION_MODEL)
    return models

def run_processing_pipeline(data):
    '''
    INPUT: list of string (corpus)\n
    OUTPUT: list of list of string (annotated and sorted)
    '''
    
    models = get_models() 
    return [process_document(doc, models) for doc in data]



def run_annotation_pipeline(data):
    '''
    INPUT: list of string (corpus)\n
    OUTPUT: list of list of list of tuple (documents as lists of annotated sentences)
    '''

    models = get_models()
    return [annotate_document(doc, models) for doc in data]


def process_document(document, models):
    '''
    INPUT:  string (document)\n
    OUTPUT: list of string (document as a list of tokens, where each token is a lemma and filtered)
    '''
    sentences = run_tokenization(document)
    processed_tokens = []

    for sentence in sentences:
        lemmas, ud_tags_list, suc_tags_list, suc_ne_list = run_tagging_and_lemmatization(
            sentence, models)

        ud_tag_list = [ud_tags[:ud_tags.find("|")] for ud_tags in ud_tags_list]
        processed_tokens.append([e for e in zip(sentence, lemmas, ud_tag_list, suc_ne_list)])

    return processed_tokens

def annotate_document(document, models):
    '''
    INPUT:  string (document)\n
    OUTPUT: list of list of tuple (document as a list of sentences, where each sentence is a list of tokens)
    '''
    sentences = run_tokenization(document)
    annotated_sentences = []

    for sentence in sentences:
        lemmas, ud_tags_list, suc_tags_list, suc_ne_list = run_tagging_and_lemmatization(
            sentence, models)

        ud_tag_list = [ud_tags[:ud_tags.find("|")] for ud_tags in ud_tags_list]
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


if __name__ == '__main__':
    main()
