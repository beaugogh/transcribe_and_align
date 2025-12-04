import sys
import re
import jieba
import nltk
nltk.download('punkt_tab')
from langdetect import detect
from rank_bm25 import BM25Okapi

# Ensure nltk tokenizer is available
nltk.download("punkt", quiet=True)


# ===========================================
# Tokenizers
# ===========================================
def tokenize_english(text):
    return nltk.word_tokenize(text)


def tokenize_chinese(text):
    return list(jieba.cut(text))


def tokenize_mixed(text):
    """Fallback tokenizer for bilingual or unknown text."""
    tokens = tokenize_chinese(text) + tokenize_english(text)
    tokens = [t.strip() for t in tokens if re.search(r"\w+", t)]
    return tokens


def choose_tokenizer(text):
    """Detect language and return appropriate tokenizer."""
    try:
        lang = detect(text)
    except:
        lang = "unknown"

    if lang == "en":
        print("🌐 Language detected: English")
        return tokenize_english
    elif lang.startswith("zh"):
        print("🌐 Language detected: Chinese")
        return tokenize_chinese
    else:
        print("🌐 Language detected: Mixed or unknown → using mixed tokenizer")
        return tokenize_mixed


# ===========================================
# Chunking
# ===========================================
def chunk_text(text, tokenizer, chunk_size=300, overlap=50):
    """
    Chunk text into overlapping word chunks.
    Tokenizer is chosen based on language detection.
    """
    words = tokenizer(text)

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


# ===========================================
# BM25 Search Engine Class
# ===========================================
class BM25SearchEngine:
    def __init__(self, text):
        print("🔍 Detecting language...")
        self.tokenizer = choose_tokenizer(text)

        print("⚙️  Chunking text...")
        self.chunks = chunk_text(text, self.tokenizer,
                                 chunk_size=300, overlap=50)
        print(f"✅ Created {len(self.chunks)} chunks.")

        print("⚙️  Tokenizing chunks...")
        tokenized_chunks = [self.tokenizer(chunk) for chunk in self.chunks]

        print("⚙️  Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_chunks)

        print("🎉 BM25 index built successfully!")

    def search(self, query, top_k=5):
        query_tokens = self.tokenizer(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            results.append({
                "chunk_id": idx,
                "score": float(score),
                "text": self.chunks[idx][:300] + "..."
            })
        return results


# ===========================================
# Interactive Search CLI
# ===========================================
def interactive_search(engine):
    print("\n🔎 Enter your query (English or Chinese). Type 'exit' to quit.")
    while True:
        query = input("\nQuery > ").strip()
        if query.lower() == "exit":
            print("👋 Goodbye!")
            break

        results = engine.search(query, top_k=5)
        print("\n========== Results ==========")
        for r in results:
            print(f"\n[Chunk {r['chunk_id']}] Score={r['score']}")
            print(r["text"])


# ===========================================
# Main Entry
# ===========================================
if __name__ == "__main__":
    filename = "/home/bo/workspace/transcribe_and_align/data/HP1/text_en/1/ch1.txt"

    print(f"📂 Loading file: {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    engine = BM25SearchEngine(text)
    interactive_search(engine)
