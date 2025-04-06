from input import content
import spacy
import subprocess

def load_spacy_model(model_name="en_core_web_lg"):
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model: {model_name}...")
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        return spacy.load(model_name)

def preprocess(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.like_num
        and token.is_alpha
        and token.pos_ in {"NOUN", "VERB", "ADJ"}
    ]

    cleaned = " ".join(tokens)
    return cleaned

nlp = load_spacy_model()
preprocessed_content = preprocess(content)
#print(f"Preprocessed content: {preprocessed_content}...") 

