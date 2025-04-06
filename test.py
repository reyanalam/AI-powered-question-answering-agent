import time
from sklearn.metrics.pairwise import cosine_similarity
from text_preprocessing import nlp
from chunking import semantic_chunking_content
import numpy as np

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

qa_test_cases = [
    {
        "query": "How can I automate the onboarding of new employees and ensure they have access to the right applications?",
        "expected_answer": "onboarding workflow create user account new employee assign add group different application"
    },
    {
        "query": "Is there a way for employees to request application access themselves with approval built in?",
        "expected_answer": "code workflow automate app access employee self service request approval workflow learn stay"
    },
    {
        "query": "Can I use workflows to manage software licenses for different departments?",
        "expected_answer": "code workflow automate app access employee self service request approval workflow learn stay"
    }
]


def cosine_sim(a, b):

    a_vec = embed_text(a).reshape(1, -1)
    b_vec = embed_text(b).reshape(1, -1)
    return cosine_similarity(a_vec, b_vec)[0][0]


def test_results_not_empty():
    print("Test: Results are not empty")
    for case in qa_test_cases:
        results = semantic_search(case["query"], semantic_chunking_content)
        assert len(results) > 0, f"FAIL: Empty results for query: {case['query']}"
    print("PASS: All queries returned non-empty results.\n")


def test_expected_answer_similarity(threshold=0.1, top_k=3):
    print(f"Test: Expected answer is semantically similar (cosine ≥ {threshold}) to top {top_k} results")
    for case in qa_test_cases:
        results = semantic_search(case["query"], semantic_chunking_content)
        similarities = [
            cosine_sim(case["expected_answer"], result[1])
            for result in results
        ]
        assert any(sim >= threshold for sim in similarities), \
            f"FAIL: No semantically similar result (cosine < {threshold}) for query: {case['query']}"
    print("PASS: All expected answers are semantically similar.\n")


def benchmark_search():
    print("Performance Benchmark:\n")
    total_time = 0
    for case in qa_test_cases:
        start = time.time()
        results = semantic_search(case["query"], semantic_chunking_content)
        elapsed = time.time() - start
        total_time += elapsed

        if results:
            top_text = results[0]
            print(f"Query: {case['query'][:50]}... | Time: {elapsed:.4f}s | Top Result: {top_text[:60]}...")
        else:
            print(f"Query: {case['query'][:50]}... | Time: {elapsed:.4f}s | No results found.")

    print(f"\nAverage Search Time: {total_time / len(qa_test_cases):.4f} seconds\n")


def run_all():
    print("\nRunning Semantic Search Tests...\n")
    test_results_not_empty()
    test_expected_answer_similarity()
    benchmark_search()


if __name__ == "__main__":
    run_all()
