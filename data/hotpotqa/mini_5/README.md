# Mini-5 HotpotQA Test Dataset

## Purpose
This is a **frozen debugging dataset** for pre-evaluation sanity testing of the Agentic GraphRAG system. It is NOT intended for performance benchmarking.

## Contents
- **5 queries** from HotpotQA distractor dev set
- **50 unique documents** (deduplicated across queries)
  - 10 supporting documents (2 per query)
  - 40 distractor documents (8 per query)

## Query Distribution
| Query ID | Type | Answer |
|----------|------|--------|
| q1 | comparison | yes |
| q2 | comparison | no |
| q3 | bridge | Chief of Protocol |
| q4 | bridge | YG Entertainment |
| q5 | bridge | 3,677 seated |

## Files
- `queries.json` - 5 queries with question, answer, type, supporting_titles
- `corpus.json` - 50 deduplicated documents with doc_id, title, text, sentences
- `mapping.json` - Per-query supporting vs distractor title mapping
- `README.md` - This file

## Usage
```bash
# Ingest corpus (run ONCE)
python ingest_mini5.py

# Run evaluation
python evaluate_mini5.py
```

## Notes
- Source: `hotpot_dev_distractor_v1.json` (indices 0, 1, 3, 5, 7)
- Each document appears only ONCE in corpus.json
- Documents may be shared across multiple queries (tracked via `for_queries`)
- This dataset is FROZEN - do not modify for debugging purposes
