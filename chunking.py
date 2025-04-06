from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing import preprocessed_content
from text_preprocessing import nlp

def semantic_chunking(text, similarity_threshold=0.15):
    doc = nlp(text)
    
    words = [token for token in doc]
    
    if not words:
        return []

    chunks = []
    current_chunk = [words[0].text]
    prev_vector = words[0].vector.reshape(1, -1)

    for token in words[1:]:
        curr_vector = token.vector.reshape(1, -1)
        similarity = cosine_similarity(prev_vector, curr_vector)[0][0]

        if similarity >= similarity_threshold:
            current_chunk.append(token.text)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [token.text]

        prev_vector = curr_vector

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Run the chunking
semantic_chunking_content = semantic_chunking(preprocessed_content)
#print(f"Semantic chunked content: {semantic_chunking_content}...")
