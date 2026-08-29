import json
from pathlib import Path
import re
from collections import Counter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "rag" / "documents"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
index = None


def split_into_chunks(document, chunk_size=120, overlap=25):
    words = document["content"].split()
    results = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        if not chunk_words:
            break

        results.append(
            {
                "title": document["title"],
                "organization": document["organization"],
                "topic": document["topic"],
                "url": document["url"],
                "content": " ".join(chunk_words)
            }
        )

        if end >= len(words):
            break

        start += chunk_size - overlap

    return results


def load_documents():
    global chunks

    chunks = []

    for file_path in DOCUMENTS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                document = json.load(file)

            required_fields = [
                "title",
                "organization",
                "topic",
                "url",
                "content"
            ]

            if not all(field in document for field in required_fields):
                print(f"Skipping invalid RAG document: {file_path.name}")
                continue

            if not document["content"].strip():
                continue

            document_chunks = split_into_chunks(document)

            chunks.extend(document_chunks)

        except Exception as error:
            print(
                f"Error loading RAG document "
                f"{file_path.name}: {error}"
            )


def build_index():
    global index

    load_documents()

    if not chunks:
        index = None
        return

    contents = [
        chunk["content"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        contents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        np.asarray(
            embeddings,
            dtype="float32"
        )
    )
def calculate_keyword_score(query, text):
    query_words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    text_words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    stop_words = {
        "a", "an", "the", "is", "are", "am", "i", "my",
        "me", "of", "to", "in", "on", "and", "or", "can",
        "could", "why", "what", "how", "with", "have", "has",
        "been", "do", "does"
    }

    query_words = [
        word for word in query_words
        if word not in stop_words
    ]

    if not query_words:
        return 0.0

    text_counter = Counter(text_words)

    matches = sum(
        1 for word in query_words
        if text_counter[word] > 0
    )

    return matches / len(set(query_words))

def retrieve_relevant_chunks(
    query,
    top_k=3,
    similarity_threshold=0.55,
    hybrid_threshold=0.35
):
    global index

    if index is None:
        build_index()

    if index is None:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    search_count = min(
        max(top_k * 5, 10),
        len(chunks)
    )

    scores, indices = index.search(
        np.asarray(
            query_embedding,
            dtype="float32"
        ),
        search_count
    )

    results = []

    for semantic_score, chunk_index in zip(
        scores[0],
        indices[0]
    ):
        if chunk_index == -1:
            continue

        chunk = chunks[chunk_index]

        keyword_score = calculate_keyword_score(
            query,
            chunk["content"]
        )

        final_score = (
            0.75 * float(semantic_score)
            + 0.25 * keyword_score
        )

        if (
            semantic_score < similarity_threshold
            and final_score < hybrid_threshold
        ):
            continue

        results.append(
            {
                "title": chunk["title"],
                "organization": chunk["organization"],
                "topic": chunk["topic"],
                "url": chunk["url"],
                "content": chunk["content"],
                "semantic_score": float(semantic_score),
                "keyword_score": float(keyword_score),
                "score": float(final_score)
            }
        )

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]