from sentence_transformers import SentenceTransformer, util
import sqlite3
import pandas as pd



### THIS SECTION GETS ALL OCCUPATIONS
connection = sqlite3.connect("Database/ONET_DATABASE.db")
cursor = connection.cursor()
query = """
    SELECT 
        oc.title
    FROM occupation_data oc
    """
cursor.execute(query)
occupation_list = cursor.fetchall()

### THIS SECTION GETS ALL MATCHINGS OF OCCUPATIONS TO SKILLS
total_list = []
for table in ('skills', 'knowledge', 'abilities', 'work_activities'):
    query = f"""
    SELECT 
        oc.title,
        
        (sk.data_value - sr.minimum) / (sr.maximum - sr.minimum),
        cmr.description
    FROM occupation_data oc
    RIGHT JOIN {table} sk
        ON oc.onetsoc_code = sk.onetsoc_code
    LEFT JOIN scales_reference sr
        ON sk.scale_id = sr.scale_id
    LEFT JOIN content_model_reference cmr
        ON sk.element_id = cmr.element_id
    WHERE sk.scale_id = 'IM'
    """
    cursor.execute(query)
    total_list.extend(cursor.fetchall())

all_data_points = pd.DataFrame(total_list, columns=["ONET_CODE", "IMPORTANCE", "DESCRIPTION"])

# Summary of how skills are matched to occupations, concerning amount of overlap...
summary = all_data_points.groupby("DESCRIPTION")["IMPORTANCE"].agg(["mean", "var", "count"])
print(len(summary[summary["count"] != 879])) #all descriptions matched to 879 of the occupations....
# TODO: Do all jobs have the same skills/knowledge/abilities/work activities? If so, that needs to be handled. We need sparsity.


quit()
### QUIT MESSAGE HERE



### EMBEDDING SECTION
import matplotlib.pyplot as plt

# Create the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert example sentence to embedding
sentence = "Built backend services for a logistics company using Python and PostgreSQL."
emb1 = model.encode(sentence)

## HERE WE LOOK AT DISTRIBUTION OF SIMILARITY OF JOB NAMES
embeddings = model.encode([ele[0] for ele in total_list])
plt.hist(model.similarity(emb1, embeddings))
plt.show()

## HERE WE LOOK DISTRIBUTION OF SCORES
dist = []
for occupation in occupation_list:
    score = 0
    score_visor = 0
    for ele in total_list:
        if ele[0] == occupation[0]:
            emb2 = model.encode(ele[3])
            score_visor += 1
            # Compute similarity
            score += model.similarity(emb1, emb2).item()
    if score_visor != 0:
        score /= score_visor
    print(occupation, score)
    dist.append(score)

plt.hist(dist)
plt.show()