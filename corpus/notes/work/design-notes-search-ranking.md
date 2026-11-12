# Design notes: ranking for the help-center search

Problem: customers search the help center with words that do not appear in the articles
("money back" instead of "refund policy"), so keyword search misses.

## Options

1. Keep BM25, add a synonyms list maintained by support. Cheap, but the list never ends.
2. Embeddings only. Finds paraphrases, but loses exact matches like error codes.
3. Hybrid: run both, merge the scores. Exact matches get a bonus on top of semantic
   similarity.

## Proposal

Option 3. Normalize both scores to the same range before merging, otherwise BM25
dominates for long queries. Evaluate on 200 real queries from the search logs, with the
article support picked as the right answer. Report the share of queries where the right
article is first, and in the top five.
