# Ingestion Pipeline Bug Fixes

## Date: 2026-01-10

## Bugs Fixed

### 1. **ConflictResolutionAgent: Property Merge Bug** (CRITICAL)
**File**: `src/agents/conflict_resolution_agent.py:281-285`

**Problem**: When merging duplicate entities, properties with null values would overwrite non-null values if entities had equal confidence scores.

**Example**: 
- Entity1: `{name: "Ed Wood", nationality: None, confidence: 0.9}`
- Entity2: `{name: "Ed Wood", nationality: "American", confidence: 0.9}`
- Result (BUG): `{name: "Ed Wood", nationality: None}` ❌

**Fix**: Always prefer non-null values over null values during merge
```python
elif merged_properties[key] is None or entity.get('confidence', 0) > cluster[0].get('confidence', 0):
    if value is not None:  # Don't overwrite with None
        merged_properties[key] = value
```

---

### 2. **ConflictResolutionAgent: Type-Agnostic Deduplication** (CRITICAL)
**File**: `src/agents/conflict_resolution_agent.py:156-164`

**Problem**: Entities with the same name but DIFFERENT types were being merged as duplicates.

**Example**: "Ed Wood" (Person) was merged with "Ed Wood" (CreativeWork/film), losing the person's nationality.

**Fix**: Check entity type before considering duplicates
```python
# CRITICAL: Check entity type first - don't merge different types!
type1 = entity1.get('type', '').lower()
type2 = entity2.get('type', '').lower()
if type1 != type2:
    return False  # Different types can't be duplicates
```

---

### 3. **EntityAgent: Temperature Too High**
**File**: `src/agents/entity_agent.py:287, 359`

**Problem**: Temperature=0.1 allows randomness in entity extraction, causing inconsistent results.

**Fix**: Changed to temperature=0.0 for deterministic extraction

---

### 4. **RelationAgent: Temperature Too High**
**File**: `src/agents/relation_agent.py:160, 278`

**Problem**: Temperature=0.1 allows randomness in relationship extraction.

**Fix**: Changed to temperature=0.0 for deterministic extraction

---

### 5. **SchemaAgent: Temperature Too High**
**File**: `src/agents/schema_agent.py:142`

**Problem**: Temperature=0.1 allows randomness in schema inference.

**Fix**: Changed to temperature=0.0 for deterministic schema inference

---

## Impact

**Before Fixes:**
- Ed Wood nationality: `None` ❌
- Scott Derrickson nationality: `"American"` ✅ (by chance)
- Inconsistent extraction across documents

**After Fixes:**
- Both entities will have correct nationality
- No cross-type merging
- Deterministic, reproducible extraction

---

## Next Steps

1. ✅ All ingestion agents audited
2. ⏳ Re-ingest HotpotQA corpus with fixed pipeline
3. ⏳ Run 100-query baseline evaluation
4. ⏳ Analyze results and compare to KG²RAG baseline

---

## Lessons Learned

1. **Always validate property merging logic** - null values need special handling
2. **Type-check before deduplication** - same name ≠ same entity
3. **Use temperature=0.0 for deterministic systems** - especially for data extraction
4. **Test with edge cases** - entities that share names across types

