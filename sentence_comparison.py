from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

past = "Built backend services for a logistics company using Python and PostgreSQL."
future = "Looking to become a data engineer focusing on cloud pipelines and big data."

# Convert to embeddings
emb1 = model.encode(past, convert_to_tensor=True)
emb2 = model.encode(future, convert_to_tensor=True)

# Compute cosine similarity
score = util.cos_sim(emb1, emb2)
print(score.item())  # Value between -1 and 1
