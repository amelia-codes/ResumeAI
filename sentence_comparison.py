from sentence_transformers import SentenceTransformer, util
import sqlite3

job_name = "math"

connection = sqlite3.connect("Database/ONET_DATABASE.db")
cursor = connection.cursor()

query = """
    SELECT 
        oc.title
    FROM occupation_data oc
    """

cursor.execute(query)
occupation_list = cursor.fetchall()


total_list = []

from pprint import pprint

for table in ('skills', 'knowledge', 'abilities', 'work_activities'):
    query = """
    SELECT 
        oc.title,
        sk.scale_id,
        (sk.data_value - sr.minimum) / (sr.maximum - sr.minimum),
        cmr.element_name
    FROM occupation_data oc
    RIGHT JOIN skills sk
        ON oc.onetsoc_code = sk.onetsoc_code
    LEFT JOIN scales_reference sr
        ON sk.scale_id = sr.scale_id
    LEFT JOIN content_model_reference cmr
        ON sk.element_id = cmr.element_id
    WHERE sk.scale_id = 'IM'
    """

    cursor.execute(query)
    total_list.extend(cursor.fetchall())

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "Built backend services for a logistics company using Python and PostgreSQL."

# Convert to embeddings
emb1 = model.encode(sentence)

for occupation in occupation_list:
    score = 0
    for ele in total_list:
        if ele[0] == occupation[0]:
            emb2 = model.encode(ele[3])

            # Compute similarity
            score += ele[2] * model.similarity(emb1, emb2)
    print(occupation, score)