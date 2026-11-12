# Semantix standard eval set

This branch holds the collection and queries that `semantix eval` downloads and runs. It has
no history in common with `main`, so cloning or installing Semantix never downloads it.

```bash
semantix eval
```

The first run downloads this branch into `~/.semantix/eval/set`. See the
[documentation](https://github.com/57Ajay/Semantix/blob/main/docs.md#measuring-search-quality)
for what the command reports and how to compare changes.

## Contents

| Path | What it is |
| --- | --- |
| `set.toml` | Name and version of the set |
| `queries.toml` | Every query with its graded answers |
| `corpus/` | The files that are searched |
| `sources.toml` | Origin, author, licence and checksum of every file that came from elsewhere |
| `licenses/` | Licence texts of the three code projects |
| `results/` | Reports from reference runs on a 4-core Xeon with AVX512-BF16: the float32 baseline and the same run in bfloat16 |

## Changing the set

Raise `version` in `set.toml` whenever files or answers change, so reports made with
different versions are never compared without a warning. Every file taken from elsewhere
needs an entry in `sources.toml` with a licence that allows redistribution (public domain,
CC0, or the project's own open source licence for code).

Look at every picture, video segment and recording before writing a query about it. Search
results for stock photos are often labelled wrongly: two of three "dog in the snow" results
were a cat, and the "spacewalk" results were people in VR headsets.
