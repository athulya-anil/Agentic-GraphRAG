# Mini-25 Gold-Only: Failed Query Analysis

**Experiment**: Gold-only corpus (50 supporting docs, 0 distractors)
**Best strategy**: HYBRID at 68% EM (17/25)

There are 9 distinct queries that fail across at least one strategy. Below is a per-query root cause analysis using the **best available result** (HYBRID where it passes, otherwise analyzing why all strategies fail).

---

## Failure Categories Summary

| Category | Queries | Count |
|---|---|---|
| **LLM Reasoning Error** | q4, q9, q19, q24 | 4 |
| **Answer Format Mismatch** | q10, q14, q15, q21 | 4 |
| **Retrieval Failure** | q11 | 1 |

---

## Category 1: LLM Reasoning Errors (4 queries)

These fail because the LLM produces the **wrong answer** despite having all the evidence.

### q4 - "Who is older, Annie Morton or Terry Richardson?"
- **Gold**: Terry Richardson
- **Predicted**: Annie Morton (vector/hybrid), unknown (graph/orchestrator)
- **Coverage**: 100% (both docs retrieved)
- **Root Cause**: **LLM comparison error**. The docs clearly state Annie Morton born Oct 8, 1970 and Terry Richardson born Aug 14, 1965. Richardson is older (born earlier). The LLM incorrectly picks Annie Morton. This is a pure reasoning failure — the LLM misinterprets "older" or confuses the dates.

### q9 - "Are Random House Tower and 888 7th Avenue both used for real estate?"
- **Gold**: no
- **Predicted**: yes (all strategies)
- **Coverage**: 100% (vector/hybrid), 50% (graph)
- **Root Cause**: **LLM semantic confusion**. Random House Tower is "mixed-use" (publisher HQ + apartments). 888 7th Avenue is an "office skyscraper" with Vornado Realty Trust HQ. The LLM likely sees "Vornado Realty Trust" (a real estate company) at 888 7th Ave and "luxury apartment complex" at Random House Tower, and concludes both involve real estate. But the gold answer is "no" — the intended distinction is that Random House Tower is used for publishing/residential, not real estate business. This requires nuanced domain understanding the LLM lacks.

### q19 - "Which performance act has a higher instrument to person ratio, Badly Drawn Boy or Wolf Alice?"
- **Gold**: Badly Drawn Boy
- **Predicted**: badly drawn boy (HYBRID only), unknown (vector/graph/orchestrator)
- **Coverage**: 100% (vector/hybrid), 50% (graph — missing Badly Drawn Boy doc)
- **Root Cause**: **Implicit reasoning required**. Wolf Alice = 4 people with guitar, guitar, bass, drums = 4 instruments = 1:1 ratio. Badly Drawn Boy = 1 person described as "multi-instrumentalist" = many instruments:1 person = higher ratio. The LLM needs to infer instrument counts from descriptions like "multi-instrumentalist" and band member lists. Vector/orchestrator fail because the LLM can't make this inference; HYBRID succeeds (likely due to graph providing structured entity info about members).

### q24 - "Kaiser Ventures corporation was founded by an American industrialist who became known as the father of modern American shipbuilding?"
- **Gold**: Henry J. Kaiser
- **Predicted**: yes (all strategies)
- **Coverage**: 100%
- **Root Cause**: **Question misclassification**. The question is phrased as a statement ending with "?" — it reads like a yes/no confirmation rather than a "who" question. The LLM interprets it as "Is it true that Kaiser Ventures was founded by..." and answers "yes" instead of extracting the name "Henry J. Kaiser". This is an ambiguous question format issue combined with the system prompt instruction "For yes/no questions, answer exactly 'yes' or 'no'."

---

## Category 2: Answer Format Mismatch (4 queries)

These fail because the LLM produces the **correct entity** but in a different format than the gold answer. The EM metric is too strict.

### q10 - "The football manager who recruited David Beckham managed Manchester United during what timeframe?"
- **Gold**: "from 1986 to 2013"
- **Predicted**: "1986 to 2013" (all strategies)
- **Coverage**: 100%
- **Root Cause**: **Missing preposition**. The LLM correctly identifies the timeframe but omits "from". The gold answer includes "from 1986 to 2013" while the prediction is "1986 to 2013". Both SubEM and strict EM fail because "from" is in the gold but not the prediction (SubEM checks if gold is in prediction, not the reverse). This is a **false negative** — the answer is semantically correct.

### q14 - "Roger O. Egeberg... a president that served during what years?"
- **Gold**: "1969 until 1974"
- **Predicted**: "19691974" (vector/hybrid/orchestrator), "nixon 19691974" (graph)
- **Coverage**: 100% (vector/hybrid), 50% (graph)
- **Root Cause**: **Number normalization issue**. The LLM outputs "1969–1974" which gets normalized to "19691974" (the dash/hyphen is removed as punctuation). The gold "1969 until 1974" normalizes to "1969 until 1974". The word "until" is the key difference — SubEM can't match "1969 until 1974" inside "19691974". The LLM correctly identifies the years but uses a dash instead of "until".

### q15 - "Which writer was from England, Henry Roth or Robert Erskine Childers?"
- **Gold**: "Robert Erskine Childers DSC"
- **Predicted**: "robert erskine childers" (all strategies)
- **Coverage**: 100%
- **Root Cause**: **Missing honorific**. The gold answer includes "DSC" (Distinguished Service Cross) which the LLM omits. The doc text says "Robert Erskine Childers DSC" but the LLM strips the honorific when answering. SubEM fails because "robert erskine childers dsc" is not found in "robert erskine childers". This is a **false negative** — the correct person is identified.

### q21 - "Who was the writer of These Boots Are Made for Walkin' and who died in 2007?"
- **Gold**: "Barton Lee Hazlewood"
- **Predicted**: "lee hazlewood" (all strategies)
- **Coverage**: 100%
- **Root Cause**: **Incomplete name extraction**. The doc says "Barton Lee Hazlewood" as his full name, but the doc title is "Lee Hazlewood" and the song doc says "written by Lee Hazlewood". The LLM returns the more commonly used name rather than the full birth name. SubEM fails because "barton lee hazlewood" is not in "lee hazlewood". This is a **false negative** — the correct person is identified.

---

## Category 3: Retrieval Failure (1 query)

### q11 - "Brown State Fishing Lake is in a country that has a population of how many inhabitants?"
- **Gold**: "9,984"
- **Predicted**: "unknown" (all strategies)
- **Coverage**: 100% (vector/hybrid), 0% (graph)
- **Root Cause**: **Terminology confusion in bridging**. The question says "country" but the answer is about Brown **County**, Kansas (population 9,984). The docs say "Brown County" not "Brown country". The LLM sees "country" in the question and looks for a nation's population rather than a county's. Even with 100% doc coverage, the LLM returns "unknown" because it doesn't equate "country" (likely a typo/error in the HotpotQA question) with "county". This may actually be an issue with the original HotpotQA question itself.

---

## Cross-Strategy Failure Patterns

| Query | Vector | Graph | Hybrid | Orchestrator | Failure Type |
|---|---|---|---|---|---|
| q4 | wrong | unknown | wrong | unknown | LLM reasoning |
| q9 | wrong | wrong | wrong | wrong | LLM reasoning |
| q10 | format | format | format | format | Format mismatch |
| q11 | unknown | unknown | unknown | unknown | LLM confusion |
| q14 | format | format | format | format | Format mismatch |
| q15 | format | format | format | format | Format mismatch |
| q19 | unknown | unknown | **PASS** | unknown | LLM reasoning (hybrid fixes) |
| q21 | format | format | format | format | Format mismatch |
| q24 | wrong | wrong | wrong | wrong | LLM reasoning |

---

## Key Insights

1. **4/9 failures are false negatives** (q10, q14, q15, q21) — the system finds the correct answer but EM matching is too strict. With a more lenient metric (F1 or human eval), these would pass. This means the **true accuracy is closer to 84% (21/25)** for HYBRID.

2. **Retrieval is not the bottleneck** in gold-only mode. Every failed query (except q11 on graph) has 100% supporting doc coverage. The failures are all in the **answering/reasoning layer**.

3. **q24 is fixable** by improving the LLM prompt to handle ambiguous question formats — the system prompt's "For yes/no questions, answer exactly 'yes' or 'no'" causes it to misclassify the question.

4. **q11 appears to be a bad question** in HotpotQA — "country" vs "county" confusion.

5. **HYBRID uniquely solves q19** — likely because combining graph entity structure (member counts) with vector text helps the LLM reason about "instrument to person ratio".

6. **Graph strategy has additional failures** due to low coverage (50% on many queries) — it fails to retrieve the second supporting doc, limiting multi-hop reasoning.
