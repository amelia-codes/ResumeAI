#from sentence_transformers import SentenceTransformer, util
import sqlite3

job_name = "math"

connection = sqlite3.connect("Database/ONET_DATABASE.db")
cursor = connection.cursor()

cursor.execute(f"SELECT onetsoc_code, title FROM occupation_data WHERE title LIKE '%{job_name}%';")
print(cursor.fetchall())

cursor.execute(f"SELECT onetsoc_code, alternate_title, short_title FROM alternate_titles WHERE alternate_title LIKE '%{job_name}%' OR short_title LIKE '%{job_name}%';")
print(cursor.fetchall())

cursor.execute(f"SELECT onetsoc_code, reported_job_title FROM sample_of_reported_titles WHERE reported_job_title LIKE '%{job_name}%';")
print(cursor.fetchall())

quit()

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "Built backend services for a logistics company using Python and PostgreSQL."
goal_words = ["developer", "python", "SQL"]

# Convert to embeddings
emb1 = model.encode(sentence, convert_to_tensor=True)
emb2 = model.encode(goal_words, convert_to_tensor=True)

# Compute similarity
score = model.similarity(emb1, emb2)
print(score)