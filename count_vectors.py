
import faiss
import pickle
import sys

try:
    index = faiss.read_index("data/faiss_index/faiss.index")
    print(f"Total vectors: {index.ntotal}")
except Exception as e:
    print(f"Error reading index: {e}")
