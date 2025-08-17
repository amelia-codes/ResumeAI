import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

def resume_phrase_scraper(resume_experience_description: str) -> list[str]:
    nlp = spacy.load("en_core_web_trf")
    tokens = nlp(resume_experience_description)
    phrases = list(tokens.noun_chunks)
    return phrases


def resume_ranker(resume_experience_phrase_list: list[str], job_description_phrase_list: list[str]) -> float:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_description_embeddings = [model.encode(phrases) for phrases in job_description_phrase_list]

    resume_experience_embeddings = [model.encode(phrases) for phrases in resume_experience_phrase_list]
    resume_scores = []
    for resume_embedding in resume_experience_embeddings:
        resume_scores.append(np.mean([model.similarity(resume_embedding, job_description_embedding) for job_description_embedding in job_description_embeddings]))

    return resume_scores
