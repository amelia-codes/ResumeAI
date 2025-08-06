from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentence = "Built backend services for a logistics company using Python and PostgreSQL."
goal_words = ["developer", "python", "SQL"]

# Convert to embeddings
emb1 = model.encode(sentence, convert_to_tensor=True)
emb2 = model.encode(goal_words, convert_to_tensor=True)

# Compute similarity
score = model.similarity(emb1, emb2)
print(score)