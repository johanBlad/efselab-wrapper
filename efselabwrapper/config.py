import os
import efselabwrapper.lemmatize as lemmatize
from efselabwrapper.tagger import SucTagger, SucNETagger, UDTagger

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def get_models():
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
