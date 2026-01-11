# HotpotQA Distractor Dataset Usage: KG²RAG vs Agentic GraphRAG

**Purpose:** Ensure we're using HotpotQA correctly for fair comparison (keeping our architecture)

---

## Critical Difference: Per-Query vs Global Indexing

### KG²RAG Approach (kg_rag_distractor.py:110-151)

```python
# For EACH query:
ctxs = sample['context']  # Get 10 documents for this query
doc_chunks = []

for ctx in ctxs:
    ent = ctx[0]  # Document title
    for i in range(len(ctx[1])):  # Each sentence
        text = f'{ent}: {ctx[1][i]}'  # Format: "Title: sentence"
        doc_chunk = TextNode(text=text, id_=f'{ent}##{str(i)}')
        doc_chunks.append(doc_chunk)

# Create fresh index for this query only
index = VectorStoreIndex(doc_chunks)
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

# Query this index
response = query_engine.query(sample_question)
```

**Key points:**
- ✅ Creates **NEW index for EACH query**
- ✅ Index contains **exactly 10 documents** (2 gold + 8 distractors)
- ✅ Distractors are query-specific (designed to confuse this particular query)
- ✅ Guarantees 100% coverage (all needed docs present)
- ❌ Doesn't test retrieval at scale (only 10 docs to search)

---

### Our Approach (ingest_hotpotqa_distractor.py)

```python
# Load ALL queries at once
queries_subset = all_queries[:100]

# Extract documents from ALL queries
title_to_text = {}
for query in queries_subset:
    context = query['context']
    for title, sentences in context:
        doc_text = ' '.join(sentences)
        if title not in title_to_text:
            title_to_text[title] = doc_text

# Result: 991 unique documents (from 100 queries × 10 docs)
documents = list(title_to_text.values())

# Ingest ALL into single shared index
pipeline.ingest_documents(documents)

# At evaluation: all 100 queries use the same index
```

**Key points:**
- ✅ Creates **ONE shared index** for all 100 queries
- ✅ Tests retrieval at scale (991 documents)
- ✅ More realistic for production RAG systems
- ⚠️ Document deduplication may affect distractor distribution
- ⚠️ Some query-specific distractors may not be present

---

## Which Approach is Correct for Distractor Setting?

### The Distractor Setting Design

From HotpotQA paper: Each query in distractor setting comes with:
- **2 gold documents** (contain the answer)
- **8 distractor documents** (topically related but don't contain answer)

**Critical insight:** The 8 distractors are **query-specific**. They're selected to be confusing for that particular query.

Example:
```
Query: "Were Scott Derrickson and Ed Wood of the same nationality?"

Gold docs (2):
  1. "Scott Derrickson" - mentions "American director"
  2. "Ed Wood" - mentions "American filmmaker"

Distractors (8):
  3. "Tim Burton" - directed Ed Wood film (confuser: related but not needed)
  4. "Johnny Depp" - starred in Ed Wood film (confuser: related but not needed)
  5. "Bela Lugosi" - acted with Ed Wood (confuser: related but not needed)
  ... 6 more distractors

The distractors are SPECIFICALLY CHOSEN to test if the system can:
- Ignore topically related but irrelevant documents
- Focus on the 2 gold documents
```

### Implications

**KG²RAG's per-query approach:**
- ✅ **Matches distractor setting design** (10 docs per query)
- ✅ Tests ability to ignore query-specific distractors
- ✅ Directly comparable to official HotpotQA leaderboard
- ❌ Doesn't test retrieval at scale

**Our global approach:**
- ✅ Tests retrieval at scale (more realistic)
- ⚠️ **Deviates from distractor setting design**
- ⚠️ May lose some query-specific distractors via deduplication
- ❌ Not directly comparable to official leaderboard

---

## Document Format Comparison

### KG²RAG

```python
# Line 129: Format each sentence as separate chunk
text = f'{ent}: {ctx[1][i]}'

# Example chunks:
"Ed Wood: Ed Wood is a 1994 American biographical period comedy-drama film."
"Ed Wood: The film concerns the period in Wood's life when he made his best-known films."
"Ed Wood: Sarah Jessica Parker, Patricia Arquette, Jeffrey Jones are among the cast."
```

**Chunking strategy:**
- ✅ One chunk per sentence
- ✅ Preserves sentence boundaries
- ✅ Chunk ID: `{title}##{sentence_index}`

---

### Our Approach

```python
# ingest_hotpotqa_distractor.py:94
doc_text = ' '.join(sentences)  # Reconstruct full document

# Then our ingestion pipeline chunks by paragraph/sentence
```

**Chunking strategy:**
- ✅ Reconstructs full document first
- ✅ Then chunks via our pipeline (paragraph boundaries)
- ⚠️ May split differently than KG²RAG

**Potential issue:** If a gold supporting fact is split across multiple chunks, we might not match their supporting fact predictions.

---

## Evaluation Metrics Comparison

### KG²RAG (hotpot_evaluate_v1.py)

```python
metrics = {
    'em': 0,           # Exact Match (answer)
    'f1': 0,           # F1 (answer)
    'prec': 0,         # Precision (answer)
    'recall': 0,       # Recall (answer)
    'sp_em': 0,        # Supporting Facts EM
    'sp_f1': 0,        # Supporting Facts F1
    'sp_prec': 0,      # Supporting Facts Precision
    'sp_recall': 0,    # Supporting Facts Recall
    'joint_em': 0,     # Answer EM × SP EM
    'joint_f1': 0,     # Combined F1
    'joint_prec': 0,   # Combined Precision
    'joint_recall': 0  # Combined Recall
}
```

**Supporting facts format:**
```python
# Line 167-168
sps = [[ent, seq] for ent, seq in source_nodes]
# Example: [["Ed Wood", 0], ["Scott Derrickson", 1]]
```

---

### Our Approach (evaluate_hotpotqa_distractor.py)

```python
metrics = {
    'em_scores': [],
    'f1_scores': [],
    'strategy_counts': {},
    'empty_contexts': 0,
}

# We compute: EM, F1 per query
# We DON'T compute: SP metrics, Joint metrics
```

**Missing metrics:**
- ❌ Supporting Facts EM
- ❌ Supporting Facts F1
- ❌ Joint EM
- ❌ Joint F1

**Why this matters:** Official HotpotQA leaderboard reports **Joint EM** as the primary metric (not just answer EM).

---

## Recommendations for Fair Comparison

### Option 1: Match KG²RAG Exactly (Apples-to-Apples)

**Change our approach to:**
1. ✅ Per-query indexing (create fresh index for each query's 10 docs)
2. ✅ Sentence-level chunking (one chunk per sentence)
3. ✅ Add supporting facts tracking
4. ✅ Add SP metrics, Joint metrics

**Pros:**
- Directly comparable to KG²RAG results
- Matches official distractor setting
- Can report to HotpotQA leaderboard

**Cons:**
- Requires modifying our ingestion pipeline
- Doesn't test retrieval at scale

---

### Option 2: Keep Our Approach, Document Differences (Recommended)

**What to do:**
1. ✅ Keep global indexing (991 docs, single index)
2. ✅ Keep our chunking strategy
3. ✅ Add supporting facts tracking
4. ✅ Add SP metrics, Joint metrics
5. ✅ **Clearly document** that we're using "distractor corpus" not "per-query distractor setting"

**Pros:**
- Tests our architecture at realistic scale
- More representative of production use
- Still uses HotpotQA data (comparable methodology)

**Cons:**
- Not directly comparable to KG²RAG's reported numbers
- Can't claim "distractor setting" (it's "distractor corpus")

**How to report:**
```
Dataset: HotpotQA dev set (100 queries)
Setting: Distractor corpus (991 unique documents from query contexts)
Note: Unlike per-query distractor setting, we index all documents globally
      to test retrieval at scale. This is more challenging than 10-doc setting.

Results:
  Answer EM: X%
  Answer F1: Y%

Comparison:
  KG²RAG (per-query, 10 docs): 60% EM
  Ours (global, 991 docs): X% EM

  Note: Not directly comparable due to different corpus sizes.
```

---

### Option 3: Run Both Approaches (Most Rigorous)

1. **Approach A:** Per-query (matches KG²RAG exactly)
2. **Approach B:** Global corpus (tests at scale)

**Report both:**
```
Distractor Setting (10 docs/query, per-query indexing):
  Answer EM: X%
  Joint EM: Y%
  Comparable to: KG²RAG 60% EM

Distractor Corpus (991 docs, global indexing):
  Answer EM: Z%
  Joint EM: W%
  Tests: Retrieval at scale
```

---

## Our Current Status

**What we're doing:**
- ✅ Global indexing (991 docs)
- ✅ Single shared index for all queries
- ✅ Answer EM, F1 metrics
- ❌ No supporting facts tracking
- ❌ No SP metrics
- ❌ No per-query indexing option

**Current ingestion:** 221/991 docs complete (~22%)

---

## Immediate Action Items

### If we want apples-to-apples comparison (Option 1):

1. **Modify ingestion script** to support per-query mode:
   ```python
   def ingest_per_query(query, pipeline):
       # Extract 10 docs for this query
       context = query['context']
       docs = [' '.join(sentences) for title, sentences in context]

       # Clear index
       pipeline.clear_index()

       # Ingest only these 10 docs
       pipeline.ingest_documents(docs)
   ```

2. **Modify evaluation script** to clear/rebuild index per query

3. **Add supporting facts tracking**

---

### If we keep our approach (Option 2 - Recommended):

1. ✅ **Keep current ingestion** (wait for 991 docs to complete)

2. **Add supporting facts tracking** to evaluation:
   ```python
   # Track which chunks contributed to answer
   supporting_facts = []
   for context in result['contexts']:
       # Extract document title and sentence ID
       supporting_facts.append([title, sentence_id])
   ```

3. **Add SP metrics** to evaluation script (copy from hotpot_evaluate_v1.py)

4. **Document methodology clearly** in results

---

## Conclusion

**For your goal (study their dataset usage to ensure correctness):**

**Key finding:** KG²RAG uses **per-query indexing** (10 docs per query), we use **global indexing** (991 docs shared).

**Recommendation:** Keep our global approach (Option 2) because:
- ✅ Tests our architecture at realistic scale
- ✅ More representative of production RAG
- ✅ Still uses HotpotQA data (valid comparison)
- ✅ Just need to document the difference clearly

**Required changes:**
1. ✅ Add supporting facts tracking (moderate effort)
2. ✅ Add SP metrics (copy their evaluation script)
3. ✅ Document that we use "distractor corpus" not "per-query distractor"

**Current baseline evaluation** (after ingestion completes) will tell us:
- How well our architecture works at scale (991 docs)
- Whether the 0% EM was due to coverage (1% with aligned data) or architecture

**Then we can decide:** If EM is still low (<10%), we might want to try per-query mode to match KG²RAG exactly.
