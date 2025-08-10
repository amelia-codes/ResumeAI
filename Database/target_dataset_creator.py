import sqlite3
import pandas as pd
import spacy

connection = sqlite3.connect("ONET_DATABASE.db")
cursor = connection.cursor()
query = """
    SELECT 
        description
    FROM content_model_reference
"""
cursor.execute(query)
occupation_list = cursor.fetchall()

print("check 1")

full_list = []
nlp = spacy.load("en_core_web_trf")
for section_text in occupation_list:
    tokens = nlp(section_text[0])
    full_list.extend(list(tokens.noun_chunks))

print("check 2")

pd.DataFrame(full_list).to_excel("target_dataset.xlsx")
