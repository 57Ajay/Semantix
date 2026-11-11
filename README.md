# Semantix

Local semantic search over your own files. Point it at directories, and later ask in plain
words: "how did I do the AWS deployment", "screenshot of the billing dashboard", "the clip
where someone gives a speech". Semantix answers with the file, the line range or the video
timestamp.

It embeds code, text, images, video and audio into one shared vector space with Google's
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), so a text query can find
any of them. Everything is stored on your machine. After the one-time model download nothing
touches the network.

```console
$ semantix search "how did I deploy to aws"
 1  0.197  shop-api/.github/workflows/deploy.yml:1-16  code · keyword match
 2  0.195  shop-api/infra/deploy.sh:1-14  code · keyword match

$ semantix search "a bee on a flower" --type video
 1  0.131  clips/holiday.mp4 @ 00:30–01:00  video
```

## Requirements

- Linux, or macOS 14 or newer on Apple Silicon (M-series)
- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/)
- Disk: 1.5 GB for the model, plus 1.3 GB of libraries with CPU-only PyTorch or about 6 GB
  with the NVIDIA build
- 2 GB of free memory for text only, 4 GB with images, video and audio
- A GPU is optional. Semantix uses CUDA if it finds it, then Apple Metal, then the CPU

## Install

```bash
uv tool install --torch-backend auto git+https://github.com/57Ajay/Semantix
```

`--torch-backend auto` makes uv pick the PyTorch build that fits the machine: the small
CPU-only build when there is no NVIDIA GPU, or the CUDA build that matches the installed
driver when there is one. Without the flag, Linux gets the newest CUDA build (about 6 GB),
which also runs on a CPU but wastes the space and needs a recent driver to use a GPU.

Or from a clone, for development:

```bash
git clone https://github.com/57Ajay/Semantix
cd Semantix
uv sync
uv run semantix --version
```

## Quick start

```bash
semantix setup
```

Setup asks which capabilities you want (text and code are always on; images, video and audio
are optional), downloads the model and starts the background service.

```bash
semantix sync ~/projects
```

The first sync of a directory shows what it found and lets you embed everything, pick file
types, or skip extensions. Nothing is embedded before you confirm.

```bash
semantix search "retry with exponential backoff"
```

```bash
semantix sync
```

Run without arguments, `sync` checks every tracked location and shows what is new, changed,
moved or deleted, with a time estimate, before doing any work.

## Commands

| Command | What it does |
| --- | --- |
| `semantix setup` | Choose capabilities, download the model, start the service |
| `semantix sync [paths]` | Embed new and changed files; no paths means everything tracked |
| `semantix search "query"` | Search; `--type`, `--in`, `--like`, `--json`, `--open` and more |
| `semantix status` | Service, model and index state |
| `semantix list` | Tracked locations |
| `semantix forget <path>` | Remove a location from the index (files are never touched) |
| `semantix check <path>` | Explain whether a path would be embedded, and why not |
| `semantix start` / `stop` / `restart` | Control the background service |
| `semantix autostart enable` / `disable` | Start the service at login |
| `semantix config` | Show or change settings |
| `semantix benchmark` | Measure speed on this machine and verify the results |

## How it works

- A background service keeps the model, the settings and the vector index in memory, so a
  query only has to embed the query text and compare vectors.
- Text and code are split into parts at natural boundaries. Videos and audio are split into
  30-second segments so a result can point at a timestamp.
- Paths are stored in full, so it does not matter which directory you run a command from, and
  a file is never stored twice.
- A changed file is detected by size, modification time and content hash, and only the parts
  that changed are embedded again. Moved and renamed files keep their vectors.
- `.gitignore` is respected. `.semantixignore` files work in any directory, and a global one
  lives at `~/.semantix/ignore`.
- All data lives under `~/.semantix`.

## Speed

Embedding speed depends almost entirely on the device.

| | Text | Image | Video | Audio |
| --- | --- | --- | --- | --- |
| 4-thread CPU | 3,600 chars/s | 3.3 s each | 0.7 frames/s | 14x realtime |
| NVIDIA Tesla T4 | 13,100 chars/s | 0.55 s each | 3.9 frames/s | 78x realtime |

A search takes 60 to 190 ms on either. On a CPU, `semantix config preset balanced` roughly
halves image and video time. Every sync shows an estimate first, based on what your machine
measured in earlier runs. More machines and the details are in
[docs.md](docs.md#results).

To measure your own machine and check that the results are correct:

```bash
semantix benchmark
```

## Documentation

[docs.md](docs.md) covers every command and setting, the ignore rules, how sync and search
behave, storage, performance and the limits of what has been tested.

## License

[MIT](LICENSE)
