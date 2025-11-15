# Synonym Handling: Approach Comparison

## Current Implementation

### ✅ What's Working (Approach 2)
**LLM generates aliases during metadata enrichment**

```python
# entity_agent.py:344-345
"3. Alternative names/synonyms (e.g., for 'Aspirin': ['medication', 'medicine', 'painkiller'])"

# Stored in Neo4j:
node_props["aliases"] = ",".join(metadata.get("aliases", []))

# Used in graph search:
WHERE toLower(n.name) CONTAINS $word
   OR toLower(coalesce(n.aliases, '')) CONTAINS $word
```

**Result**: "medications" now matches Drug entities with alias "medication"

### ⚠️ What's NOT Working Yet
**Query parser not integrated**

Created `query_parser_agent.py` but never integrated into retrieval pipeline.
Current retrieval still uses keyword matching, not intent-based parsing.

## Two Complementary Solutions

### 1. Entity-Level Synonyms (Already Working!)
**When**: During document ingestion  
**How**: LLM enriches each entity with aliases  
**Storage**: Neo4j node properties  
**Use**: Improve seed entity matching  

**Example**:
```
Entity: "Aspirin"
Aliases: ["medication", "medicine", "painkiller", "pain reliever"]
Query: "what medications treat pain"
Match: ✅ "medications" matches "medication" in aliases
```

**Limitations**:
- Only helps find entities that exist
- Doesn't solve reverse query direction
- Doesn't understand query structure

### 2. Query Intent Parser (Created but NOT Integrated)
**When**: During query execution  
**How**: LLM analyzes query to extract intent  
**Storage**: None (dynamic)  
**Use**: Generate optimal Cypher based on query structure  

**Example**:
```
Query: "Which drugs treat diabetes?"
LLM Analysis:
  - Looking for: Drug entities
  - Filtering by: "diabetes" 
  - Relationship: TREATS
  - Direction: REVERSE (from Disease to Drug)
Generated Cypher:
  MATCH (disease:Disease)<-[:TREATS]-(drug:Drug)
  WHERE disease.name CONTAINS "diabetes"
```

**Benefits**:
- Solves reverse query problem
- No storage needed
- Works with any phrasing
- Understands query intent

## Recommendation: Use BOTH!

### Why Both Are Needed:

**Entity Aliases** (Approach 2) solve:
✅ "medications" → finds entities with alias "medication"  
✅ "meds" → finds entities with alias "meds"  
✅ Fast (no per-query LLM call)

**Query Parser** (Approach 1) solves:
✅ Reverse queries: "Which drugs treat X?"  
✅ Complex queries: "Who founded companies that manufacture drugs?"  
✅ Direction inference: forward vs reverse  
✅ Optimal Cypher generation

### Integration Plan:

```python
def retrieve_graph(query, params):
    # Step 1: Parse query intent with LLM
    parser = get_query_parser(schema=self.neo4j_manager.get_schema())
    intent = parser.parse_query(query)
    
    # Step 2: Generate optimized Cypher
    cypher, cypher_params = parser.construct_cypher_query(intent)
    
    # Step 3: Execute (will use aliases for entity matching)
    results = self.neo4j_manager.execute_query(cypher, cypher_params)
    
    return results
```

## What We Have vs What We Need

| Component | Status | Location |
|-----------|--------|----------|
| Entity aliases generation | ✅ Working | entity_agent.py:344-378 |
| Store aliases in Neo4j | ✅ Working | ingestion.py:247 |
| Search aliases in queries | ✅ Working | retrieval.py:277-280 |
| Query intent parser | ✅ Created | query_parser_agent.py |
| Parser integration | ❌ Missing | retrieval.py (needs update) |
| Hard-coded synonyms | ✅ Removed | (deleted config file) |

## Next Steps

1. ✅ Remove hard-coded synonym file  
2. ⏭️ Integrate query parser into retrieval pipeline  
3. ⏭️ Test combined approach  
4. ⏭️ Measure improvement on reverse queries

**Expected Results**:
- Simple queries: Fast (use aliases)  
- Complex/reverse queries: Accurate (use parser)  
- Best of both worlds!
