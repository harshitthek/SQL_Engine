# Text-to-SQL Evaluation & Error Analysis Report

**Date/Time:** 2026-09-08 10:00:05 UTC  
**Benchmark Split:** Spider Validation Set (`dev.json`)  
**Total Examples Evaluated:** 20  
**Overall Exact Match (EM):** 80.00% (16/20)  
**Overall Execution Accuracy (EX):** 85.00% (17/20)  

---

## 1. Performance by SQL Complexity Tier

Evaluation results segmented by Spider difficulty tiers (simple, medium, hard, extra-hard):

| Complexity Tier   |   Total |   EM Correct | EM (%)   |   EX Correct | EX (%)   |
|-------------------|---------|--------------|----------|--------------|----------|
| Hard              |       2 |            1 | 50.00%   |            1 | 50.00%   |
| Medium            |      14 |           11 | 78.57%   |           12 | 85.71%   |
| Simple            |       4 |            4 | 100.00%  |            4 | 100.00%  |

---

## 2. Execution Error Breakdown

Categorization of all query execution outcomes across the benchmark:

| Error Category   |   Count | % of Total   |
|------------------|---------|--------------|
| RuntimeError     |       2 | 10.00%       |
| ResultMismatch   |       1 | 5.00%        |

---

## 3. Failure Pattern Distribution

Automated semantic diagnosis of failed queries:

| Failure Pattern                                       |   Failure Count | % of Failures   |
|-------------------------------------------------------|-----------------|-----------------|
| Schema Hallucination (Invalid Table Name)             |               1 | 33.33%          |
| Schema Hallucination (Invalid Column Name)            |               1 | 33.33%          |
| Aggregation Operator Mismatch (e.g. COUNT vs SUM/AVG) |               1 | 33.33%          |

---

## 4. Deep-Dive: Top 3 Failure Patterns

### Pattern #1: Schema Hallucination (Invalid Table Name)
- **Occurrence Count:** 1 (33.33% of all failures)

#### Representative Examples & Diagnostics:

**Example 1.1 (DB: `concert_singer`, Tier: `medium`):**
- **Question:** *"Show the name and the release year of the song by the youngest singer."*
- **Gold SQL:**
  ```sql
  SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
  ```
- **Model SQL:**
  ```sql
  SELECT T2.name ,  T2.song_release_year FROM singer AS T1 JOIN song AS T2 ON T1.singer_id  =  T2.singer_id ORDER BY T1.age LIMIT 1
  ```
- **Error Diagnosis:** `RuntimeError` — no such table: song

#### Root Cause & Mitigation:
- **Root Cause:** Schema hallucination: selecting column names that exist in other tables or inventing column synonyms not present in the SQLite catalog.
- **Mitigation Strategy:** Constrain token generation to valid identifiers present in the schema DDL using schema-guided prefix masking / grammar-based sampling.

### Pattern #2: Schema Hallucination (Invalid Column Name)
- **Occurrence Count:** 1 (33.33% of all failures)

#### Representative Examples & Diagnostics:

**Example 2.1 (DB: `concert_singer`, Tier: `hard`):**
- **Question:** *"List all song names by singers above the average age."*
- **Gold SQL:**
  ```sql
  SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)
  ```
- **Model SQL:**
  ```sql
  SELECT T2.Song_Name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.Singer_ID  =  T2.Singer_ID WHERE T1.Age  >  (SELECT avg(Age) FROM singer)
  ```
- **Error Diagnosis:** `RuntimeError` — no such column: T2.Song_Name

#### Root Cause & Mitigation:
- **Root Cause:** Schema hallucination: selecting column names that exist in other tables or inventing column synonyms not present in the SQLite catalog.
- **Mitigation Strategy:** Constrain token generation to valid identifiers present in the schema DDL using schema-guided prefix masking / grammar-based sampling.

### Pattern #3: Aggregation Operator Mismatch (e.g. COUNT vs SUM/AVG)
- **Occurrence Count:** 1 (33.33% of all failures)

#### Representative Examples & Diagnostics:

**Example 3.1 (DB: `concert_singer`, Tier: `medium`):**
- **Question:** *"What is the maximum capacity and the average of all stadiums ?"*
- **Gold SQL:**
  ```sql
  select max(capacity), average from stadium
  ```
- **Model SQL:**
  ```sql
  SELECT max(capacity) ,  avg(capacity) FROM stadium
  ```
- **Error Diagnosis:** `ResultMismatch` — Result mismatch: pred returned 1 rows, gold returned 1 rows

#### Root Cause & Mitigation:
- **Root Cause:** Complex compositional logic discrepancy between model interpretation and gold annotation.
- **Mitigation Strategy:** Add targeted few-shot demonstrations for complex compositional SQL patterns.

---

## 5. Summary & Engineering Recommendations

1. **Schema Linking & Foreign Key Enforcement:** The primary source of execution errors is multi-table join confusion. Enhancing schema serialization with explicit `FOREIGN KEY (a) REFERENCES b(c)` edges improves join path discovery.
2. **Grammar-Based Constrained Decoding:** Enforcing SQLite grammar at decode time eliminates 100% of syntax errors and prevents schema hallucination.
3. **Execution-Guided Self-Correction / Re-ranking:** Use the SQLEvaluator in a test-time beam re-ranking loop: generate 4 candidates, execute against SQLite, reject queries that produce syntax or runtime errors, and select the highest-scoring valid query.
