import spacy

def job_description_phrase_scraper(job_description: str) -> list[str]:
    nlp = spacy.load("en_core_web_trf")
    tokens = nlp(job_description)
    phrases = list(tokens.noun_chunks)
    return phrases