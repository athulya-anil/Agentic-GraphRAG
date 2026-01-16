# Mini-5 HotpotQA Evaluation Report

**Date:** 2026-01-16
**Dataset:** mini_5 (5 queries, 50 documents)
**Purpose:** Pre-evaluation sanity test

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Overall EM | **1/5 (20%)** |
| Comparison EM | 1/2 (50%) |
| Bridge EM | 0/3 (0%) |
| Avg Supporting Coverage | **0%** |

**Verdict:** Critical issues detected. The system is not retrieving supporting documents.

---

## Per-Query Results

### Q1: Comparison (nationality)
- **Question:** Were Scott Derrickson and Ed Wood of the same nationality?
- **Expected:** yes
- **Predicted:** no
- **EM:** ✗
- **Strategy:** GRAPH (correct routing)
- **Issue:** Failed to find "Scott Derrickson" and "Ed Wood" entities in graph

### Q2: Comparison (location)
- **Question:** Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?
- **Expected:** no
- **Predicted:** no
- **EM:** ✓
- **Strategy:** GRAPH (correct routing)
- **Note:** Correct answer but 0% supporting coverage (lucky guess via default)

### Q3: Bridge (government position)
- **Question:** What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
- **Expected:** Chief of Protocol
- **Predicted:** meet corliss archer
- **EM:** ✗
- **Strategy:** GRAPH (correct routing)
- **Issue:** AnswerSelector returned entity name instead of attribute

### Q4: Bridge (formed by)
- **Question:** 2014 S/S is the debut album of a South Korean boy group that was formed by who?
- **Expected:** YG Entertainment
- **Predicted:** 2014
- **EM:** ✗
- **Strategy:** GRAPH (correct routing)
- **Issue:** AnswerSelector extracted year from album title instead of traversing to organization

### Q5: Bridge (seating capacity)
- **Question:** The arena where the Lewiston Maineiacs played their home games can seat how many people?
- **Expected:** 3,677 seated
- **Predicted:** maine
- **EM:** ✗
- **Strategy:** VECTOR (misrouted - should be GRAPH for multi-hop)
- **Issue:** Misrouted to VECTOR; AnswerSelector returned location instead of capacity

---

## Root Cause Analysis

### 1. Supporting Document Coverage: 0%

**Critical Finding:** The graph traversal queries are not finding the supporting documents by title.

**Cause:** The ingested entities don't preserve document titles as searchable node names. When searching for "Scott Derrickson" or "Ed Wood", the graph search returns no matches because entities were extracted with different naming conventions.

**Evidence from logs:**
```
Missing: ['Scott Derrickson', 'Ed Wood']
Missing: ['Laleli Mosque', 'Esma Sultan Mansion']
Missing: ['Shirley Temple', 'Kiss and Tell (1945 film)']
```

### 2. AnswerSelector Failures

The AnswerSelector is extracting wrong entity types:
- Q3: Returned "meet corliss archer" (a related show) instead of "Chief of Protocol" (the position)
- Q4: Returned "2014" (album year) instead of "YG Entertainment" (organization)
- Q5: Returned "maine" (location) instead of "3,677 seated" (capacity)

**Root Cause:** The AnswerSelector matches KG node names to the query but lacks query-type-aware filtering. It needs to understand what *type* of entity the question is asking for.

### 3. Query Routing

- 4/5 queries routed to GRAPH (mostly correct)
- Q5 was misrouted to VECTOR (should be GRAPH for "arena where X played → capacity")

---

## Storage Statistics

| Store | Count |
|-------|-------|
| Neo4j Nodes | 354 |
| Neo4j Edges | 340 |
| FAISS Vectors | 577 |
| Documents Ingested | 50 |

---

## Recommendations

### Immediate Fixes

1. **Entity-Document Linking:** Ensure each entity node has a `source_title` property matching the original document title for coverage tracking.

2. **AnswerSelector Query-Type Awareness:** Add logic to filter candidate answers by expected entity type:
   - COMPARISON → Boolean or attribute comparison
   - BRIDGE "what position" → Role/Position entity
   - BRIDGE "formed by who" → Organization/Person entity
   - BRIDGE "how many" → Numeric value

3. **Query Routing for Numeric Queries:** Questions containing "how many" should route to GRAPH to find numeric attributes.

### Investigation Needed

1. Why are supporting document titles not matching graph entity names?
2. Are entities being normalized differently during ingestion vs. retrieval?
3. Is the graph schema capturing the right relationships for multi-hop traversal?

---

## Conclusion

The mini-5 evaluation reveals critical issues in the retrieval pipeline:

1. **0% supporting coverage** indicates the graph cannot find relevant entities by title
2. **AnswerSelector** is selecting wrong entity types without query-type filtering
3. **Bridge questions (0% EM)** are fundamentally broken due to missing multi-hop traversal

Before running full HotpotQA evaluation, these issues must be resolved. The current system cannot effectively answer multi-hop questions.
