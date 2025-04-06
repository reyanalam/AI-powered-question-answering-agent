import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing import nlp
from chunking import semantic_chunking_content

def embed_text(text):
    doc = nlp(text)
    
    vectors = [token.vector for token in doc if token.has_vector and not token.is_stop and not token.is_punct]
    if vectors:
        return np.mean(vectors, axis=0).reshape(1, -1)
    else:
        return np.zeros((1, nlp.vocab.vectors_length)) 
    
def semantic_search(query, chunks, top_k=3, similarity_threshold=0.6):
    query_vector = embed_text(query)
    chunk_vectors = [embed_text(chunk) for chunk in chunks]

    similarities = [cosine_similarity(query_vector, chunk_vec)[0][0] for chunk_vec in chunk_vectors]
    
    top_indices = np.argsort(similarities)[::-1]
    
    results = []
    print(f"\n Relevant Chunks for Query:\n\"{query}\"\n")

    for idx in top_indices:
        if similarities[idx] >= similarity_threshold:
            chunk = chunks[idx]
            score = similarities[idx]
            results.append((chunk, score))
            if len(results) == top_k:
                break
    
    if not results:
        print("Sorry, I couldn't find any relevant information.")
        return []

    # Pretty print results
    for i, (chunk, score) in enumerate(results, 1):
        print(f"{i}️⃣ Similarity: {score:.2f}\n\"{chunk}\"\n")

    return [chunk for chunk, _ in results]


semantic_search_query = input("Enter your query: ").strip()
semantic_search_results = semantic_search(semantic_search_query, semantic_chunking_content)
