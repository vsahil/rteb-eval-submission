# RTEB Data Format

RTEB evaluates retrieval tasks. Each task is represented as a directory with
queries, corpus documents, and relevance judgments.

```text
<data_path>/<task_name>/
  queries.jsonl
  corpus.jsonl
  relevance.jsonl
  custom_instruction.json  # optional
```

`<data_path>` is the root directory passed to the evaluator. `<task_name>` is a
dataset or subtask name, such as `AILA_casedocs`, `financebench`, or an
agentic-code repository/task shard.

## Queries

`queries.jsonl` contains one JSON object per query.

```json
{"id": "q1", "text": "What statute governs this legal scenario?"}
{"id": "q2", "text": "Which file should be edited to fix this issue?"}
```

Required fields:

- `id`: stable string query identifier.
- `text`: query text to embed.

## Corpus

`corpus.jsonl` contains one JSON object per retrievable document.

```json
{"id": "d1", "text": "Document text, passage text, code file content, or answer text."}
{"id": "d2", "text": "Another retrievable document."}
```

Required fields:

- `id`: stable string document identifier.
- `text`: document text to embed.

Optional fields:

- `title`: document title. If present, evaluators may prepend it to `text`.
- `metadata`: object with non-scoring metadata.
- `chunks`: list of chunk objects for chunked document embedding. Each chunk
  should have `id` and `text`.

## Relevance

`relevance.jsonl` contains query-to-document relevance labels. Each line maps
one or more query ids to document-score maps.

```json
{"q1": {"d1": 1, "d2": 0}}
{"q2": {"d2": 1}}
```

Scores are integers. A score greater than zero is considered relevant for
Recall. NDCG uses the provided score as graded relevance.

For small tasks, `relevance.json` may be used instead:

```json
{
  "q1": {"d1": 1, "d2": 0},
  "q2": {"d2": 1}
}
```

## Optional Instructions

`custom_instruction.json` can override the default query and corpus instruction
prefixes.

```json
{
  "query": "Represent the query for retrieving supporting documents: ",
  "corpus": "Represent the document for retrieval: "
}
```

If omitted, the evaluator uses the default retrieval instructions.

## Public And Private Tasks

The same format applies to all public RTEB tasks and to server-side private
tasks. Private datasets are not released in this repository; they are evaluated
by the RTEB organizers using the same query/corpus/relevance abstraction.

For public reproduction, datasets can be supplied in this local format or
downloaded/converted by dataset-specific adapters. The release repository should
not include private data.

The included `rteb.agentic_coding_dataset` loader targets local agentic-coding
retrieval tasks in this format. Loaders for non-agentic text benchmarks are
implemented separately.

## Expected Outputs

The evaluator writes:

```text
<output_path>/<task_name>/
  retrieve_pred.json
  retrieve_eval.json
```

`retrieve_pred.json` maps each query id to ranked document scores:

```json
{
  "q1": {"d1": 0.92, "d2": 0.31}
}
```

`retrieve_eval.json` contains retrieval metrics such as `ndcg_at_10` and
`recall_at_100`.
