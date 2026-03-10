from word2vec import Word2VecSGNS
def most_similar(word, model, k=5):
    word = word.lower()
    if not model.contains(word):
        raise ValueError("Word not in vocabulary")

    embeddings = model.get_embedding_matrix()
    vocab = model.get_vocab()

    #calculate cosine similarities
    word_vec = model.get_embedding(word)
    word_norm = np.linalg.norm(word_vec)
    norms = np.linalg.norm(embeddings, axis=1)
    sims = np.dot(embeddings, word_vec) / (norms * word_norm + 1e-10)
    similar_ids = np.argsort(-sims)

    results = []
    for idx in similar_ids:
        similar_word = vocab[idx]
        similarity = sims[idx]
        if similar_word == word:
            continue
        results.append(similar_word)
        if len(results) >= k:
            break

    return results

with open("Alice.txt", "r", encoding="utf8") as f:
    text = f.read()

model = Word2VecSGNS(
    text=text,
    embedding_dim=50,
    lr=0.025,
    neg_samples=15,
    subsample_t=1e-3
)

model.train(epochs=5)

query_word = 'alice'
print(f"\nNearest words to '{query_word}':")
print(most_similar(query_word, model))