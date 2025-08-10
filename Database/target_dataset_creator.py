import sqlite3
import pandas as pd
import spacy

connection = sqlite3.connect("ONET_DATABASE.db")
cursor = connection.cursor()
occupation_list = []
for table in ('skills', 'knowledge', 'abilities', 'work_activities'):
    query = f"""
    SELECT
        cmr.description
    FROM occupation_data oc
    RIGHT JOIN {table} sk
        ON oc.onetsoc_code = sk.onetsoc_code
    LEFT JOIN scales_reference sr
        ON sk.scale_id = sr.scale_id
    LEFT JOIN content_model_reference cmr
        ON sk.element_id = cmr.element_id
    WHERE sk.scale_id = 'IM'
    AND sk.data_value > 2
    """
    cursor.execute(query)
    occupation_list.extend(cursor.fetchall())

print("check 1")

full_list = []
nlp = spacy.load("en_core_web_trf")
for i in range(0, len(occupation_list), 5):
    tokens = nlp("\n".join(section_text[0] for section_text in occupation_list[i: min([i+5, len(occupation_list)])]))
    full_list.extend([" ".join(token.text for token in noun_chunk) for noun_chunk in tokens.noun_chunks])
  

print("check 2")

pd.DataFrame(full_list).to_excel("target_dataset.xlsx")
