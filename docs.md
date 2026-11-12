# Semantix documentation

- [Concepts](#concepts)
- [Installing](#installing)
- [Setup](#setup)
- [Command reference](#command-reference)
- [What gets embedded](#what-gets-embedded)
- [Ignore rules](#ignore-rules)
- [How sync behaves](#how-sync-behaves)
- [How search behaves](#how-search-behaves)
- [Settings](#settings)
- [The background service](#the-background-service)
- [Storage and privacy](#storage-and-privacy)
- [Performance and memory](#performance-and-memory)
- [Benchmark and GPU testing](#benchmark-and-gpu-testing)
- [Measuring search quality](#measuring-search-quality)
- [Architecture](#architecture)
- [Development](#development)
- [Limits and untested areas](#limits-and-untested-areas)

## Concepts

**Service.** A background process that holds the model, the settings and the vector index in
memory. Every command talks to it over a local Unix socket. It is started by `semantix start`,
or automatically by the first command that needs it.

**Tracked location.** A directory or single file you have synced. Semantix remembers it, along
with the choices you made for it (which file types, which extensions to skip, which patterns
to exclude). `semantix list` shows them.

**Part.** The unit that gets one vector. A text file is split into several parts, an image is
one part, a video or audio file is one part per segment. Search results are parts, which is
why a result carries a line range or a timestamp.

**Home.** `~/.semantix`, where everything lives. Set the `SEMANTIX_HOME` environment variable
to use another directory.

## Installing

```bash
uv tool install --torch-backend auto git+https://github.com/57Ajay/Semantix
```

PyTorch is by far the largest dependency, and which build you get decides both the disk use
and whether a GPU can be used.

| Machine | With `--torch-backend auto` | Without the flag |
| --- | --- | --- |
| Linux, no NVIDIA GPU | CPU-only build, 1.3 GB of libraries in total | Newest CUDA build, about 6 GB, runs on the CPU |
| Linux, NVIDIA GPU | CUDA build matching the installed driver | Newest CUDA build, which needs a recent driver |
| macOS, Apple Silicon | The one macOS build, which includes Apple Metal | The same build |

PyTorch publishes a single macOS build, for Apple Silicon on macOS 14 or newer. There is no
separate CPU-only build for the Mac, so the flag makes no difference there. Intel Macs are not
supported.

uv marks `--torch-backend` as experimental. To choose by hand, pass a value such as `cpu` or
`cu126` instead of `auto`, or set the `UV_TORCH_BACKEND` environment variable.

To switch builds later, for example after adding a GPU, install again with `--force`:

```bash
uv tool install --force --torch-backend auto git+https://github.com/57Ajay/Semantix
```

The index and the downloaded model live in `~/.semantix` and are not affected.

## Setup

```bash
semantix setup
```

Setup walks through four choices and then starts the service.

1. **Capabilities.** Text and code are always on. Images, video and audio are optional. The
   model ships as one 1.5 GB file, so this choice does not change the download. It changes how
   much is loaded into memory: the image and audio encoders are simply left out when you do
   not want them. Video uses the image encoder.
2. **Speed preset** (only asked when no GPU is found and images or video are on). See
   [presets](#presets).
3. **Vector size.** 768 is the model's native size. 512, 256 and 128 store less at some cost
   in quality; 128 is only reasonable for text-only use.
4. **Autostart.** Whether to start the service at login.

Everything can be changed later with `semantix config`. For scripts, setup also runs without
questions:

```bash
semantix setup --yes --capabilities text,image --dimension 768 --preset balanced --no-autostart
```

### Device

`model.device` is `auto` by default: CUDA if available, then Apple Metal (MPS), then CPU. On
CUDA the model runs in bfloat16 on GPUs that accelerate it (compute capability 8.0 and up,
roughly the RTX 30 series and newer) and in float32 on older ones such as the T4, where
bfloat16 measured slower. Apple Metal and CPU use float32. float16 is never used because the
model produces invalid output in it.

`model.dtype` overrides the number format: `auto`, `float32` or `bfloat16`. bfloat16 halves
the memory for the weights. Whether it is faster depends on the hardware, and
`semantix benchmark` measures both and tells you.

## Command reference

Every command accepts `--help`.

### `semantix sync [PATHS...]`

Embeds new and changed files.

- With paths: each path is resolved to its full location first. A path inside an already
  tracked location is treated as part of it. Anything else becomes a new tracked location.
- Without paths: checks every tracked location.

In both cases it shows a plan and waits for your answer before embedding.

| Option | Meaning |
| --- | --- |
| `--only code,image` | Only these types: `code`, `text`, `image`, `video`, `audio` |
| `--skip-ext .csv,.log` | Leave out these extensions |
| `--exclude 'pattern'` | Ignore pattern, same syntax as `.gitignore`; repeatable |
| `--yes`, `-y` | Embed everything in the plan without asking |
| `--dry-run` | Show the plan and stop |
| `--force` | Embed again even if nothing changed |
| `--skip-large` | Leave out files above the size limits |

`--only`, `--skip-ext` and `--exclude` are remembered for the tracked location, so later syncs
apply them without being told again. To change them later, pass them again while syncing the
location's own path.

### `semantix search QUERY`

| Option | Meaning |
| --- | --- |
| `--limit N`, `-n N` | Number of results (default 10) |
| `--type code,image`, `-t` | Only these file types |
| `--in PATH` | Only results under this directory or file |
| `--like FILE` | Find things similar to a file instead of using a text query |
| `--mode hybrid\|vector\|keyword` | See [how search behaves](#how-search-behaves) |
| `--per-file N` | Maximum results from one file (default 2, 0 for no cap) |
| `--min-score X` | Hide results below this score |
| `--lines N`, `-l N` | Lines of text to preview per result (default 6, 0 for none) |
| `--paths` | Print only `path:lines`, one per line |
| `--json` | Machine readable output, including the full text of each part |
| `--open`, `-o` | Open the best match with the system's default application |

### `semantix status`

Shows whether the service is running, the model state and device, the capabilities, the index
contents per file type, files that are waiting or failed, and a running sync if there is one.
Works when the service is stopped. `--json` prints everything.

### `semantix list`

Tracked locations with file counts, waiting files, the last sync time and remembered rules.

### `semantix forget PATH`

Removes a location from the index. Your files are never modified.

- If the path is a tracked location, it stops being tracked.
- If the path is inside one, its files are removed and the path is added to that location's
  excludes, so the next sync does not bring it back.

### `semantix check PATH`

Explains whether a path would be embedded and, if not, which rule or setting leaves it out and
which file that rule comes from.

### `semantix start`, `stop`, `restart`

`start` launches the service in the background and returns once the model is loaded. `stop`
ends it and frees its memory. `start --foreground` runs it in the terminal, which is what the
autostart units use.

### `semantix autostart enable|disable|status`

See [the background service](#the-background-service).

### `semantix benchmark`

Measures embedding speed on this machine and checks the results. See
[benchmark and GPU testing](#benchmark-and-gpu-testing).

| Option | Meaning |
| --- | --- |
| `--device cuda\|mps\|cpu` | Test a specific device instead of the automatic choice |
| `--dtype float32,bfloat16` | Number formats to test |
| `--workload small\|large` | Force a workload size; the default is small on a CPU, large on a GPU |
| `--quick` | Same as `--workload small` |
| `--text-batch N`, `--image-batch N` | Batch sizes to test instead of the device defaults |
| `--output FILE`, `-o` | Where to write the JSON report |

### `semantix eval [SET]`

Measures search quality on a fixed collection with known answers. See
[measuring search quality](#measuring-search-quality).

| Option | Meaning |
| --- | --- |
| `--mode hybrid\|vector\|keyword\|all` | Search mode to measure (default hybrid) |
| `--split dev\|test\|all` | Which queries to run (default all) |
| `--config key=value` | Change a setting for this run only; repeatable |
| `--keyword-bonus X` | Weight of keyword matches in hybrid mode (default 0.08) |
| `--no-calibration` | Rank by raw similarity instead of calibrated scores |
| `--compare FILE` | Compare with an earlier report; `last` means the previous run |
| `--unjudged` | List top results that have no grade yet |
| `--update` | Download the newest standard set before running |
| `--device cuda\|mps\|cpu` | Run on a specific device |
| `--output FILE`, `-o` | Also write the JSON report here |

### `semantix config`

| Form | Meaning |
| --- | --- |
| `semantix config` | Show every setting, its value and meaning |
| `semantix config get KEY` | Print one value |
| `semantix config set KEY VALUE` | Change one value |
| `semantix config preset NAME` | Apply `quality`, `balanced` or `fast` |
| `semantix config path` | Print where settings, ignore rules, index, model and log live |

A running service picks changes up immediately. Changing capabilities or the device makes it
reload the model.

## What gets embedded

### File types

| Type | Recognised by |
| --- | --- |
| `code` | Source, markup and configuration extensions (`.py`, `.ts`, `.go`, `.rs`, `.sql`, `.yaml`, `.tf`, `.json`, `.html` and many more), names such as `Dockerfile` and `Makefile`, and any text file starting with `#!` |
| `text` | `.md`, `.txt`, `.rst`, `.org`, `.tex`, `.csv`, `.log` and similar, names such as `README`, and any other file that looks like UTF-8 text |
| `image` | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`, `.tif`, `.tiff` |
| `video` | `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.m4v`, `.wmv`, `.flv`, `.mpg`, `.mpeg` |
| `audio` | `.mp3`, `.wav`, `.flac`, `.ogg`, `.opus`, `.m4a`, `.aac`, `.wma` |

Files with an unknown extension are sniffed: if the first 4 KB are valid text they are treated
as text, otherwise they are skipped as unsupported. Jupyter notebooks are reduced to the
source of their cells.

### Text and code

A file is split into parts of roughly `index.chunk_chars` characters (default 1600). Splits
prefer natural boundaries: an unindented line after a blank line (a new function, class or
paragraph) or a Markdown heading. Each part is embedded with its path as a title, which the
model is trained to use, and stored with its line range.

Because the boundaries follow the content, editing one function usually changes one or two
parts. Unchanged parts keep their vectors.

### Images

One part per image. Animated images use their first frame. `image.tokens` sets how much detail
the model sees (default 280).

### Video

A video is sampled at `video.fps` frames per second (default 1) and cut into segments of
`video.segment_seconds` (default 30). Each segment becomes one part with its start and end
time. If audio is enabled and the video has a soundtrack, the soundtrack is embedded too, as
separate parts with the same timestamps, so a query can match what is said as well as what is
shown. Results from the soundtrack are labelled `soundtrack`.

The model reads at most 8,192 tokens at once, and every frame costs `video.frame_tokens` of
them. A segment is therefore limited to 109 frames at 70 tokens, 56 at 140 and 28 at 280. If
your segment length and frame rate ask for more, the segment is shortened to fit, so at 280
tokens and 1 frame per second a segment is 28 seconds, not 30.

### Audio

Cut into segments of `audio.segment_seconds` (default 30), converted to mono 16 kHz. Silent
segments and tails shorter than one second are dropped.

### Duplicates

An image, video or audio file with exactly the same content as one already embedded reuses its
vectors and is not embedded again.

## Ignore rules

Rules use `.gitignore` syntax. They are applied in layers, and a later layer overrides an
earlier one:

1. **Built-in rules.** Dependency and build directories (`node_modules`, `.venv`,
   `__pycache__`, `dist`, `build`, `target` and similar), version control directories, lock
   files, minified files, archives, binaries, model weights, databases, and secrets (`.env`,
   `*.pem`, `*.key`, SSH private keys).
2. **`.gitignore` files**, including those in parent directories up to the top of the
   repository. Turn this off with `semantix config set index.respect_gitignore false`.
3. **The global file**, `~/.semantix/ignore`.
4. **`.semantixignore` files**, in the directory being scanned, any directory below it, and
   any directory above it.
5. **`--exclude` patterns** remembered for the tracked location.

Use `!pattern` to bring something back. For example, if git ignores a `screenshots/` folder
that you do want searchable, put this in a `.semantixignore` next to it:

```gitignore
!screenshots/
```

A directory that is ignored is not entered, so to bring back something inside an ignored
directory, bring back the directory itself.

**Naming a file directly overrides the rules.** `semantix sync notes/private.log` embeds that
file even if a rule ignores `*.log`, and remembers it so later syncs keep it. `semantix forget`
undoes that.

`semantix check PATH` tells you which layer and which file decided.

## How sync behaves

### The plan

Before embedding anything, sync scans and prints a plan per tracked location:

- files per type, split into new and changed, with size and estimated time
- how many were deleted from disk, moved or renamed, or are unchanged
- what was skipped and why (ignored, unsupported type, type not selected)
- files above the size limits, listed individually

Scanning only reads file sizes and timestamps, so it is quick. Embedding is the expensive
step, and it always waits for your confirmation unless you pass `--yes`.

For a new location you are first asked what to embed: everything, chosen file types, or
everything except chosen extensions. Then you choose between embedding everything in the plan,
picking which groups to embed now, or not now.

### Change detection

| Situation | What happens |
| --- | --- |
| Size and modification time unchanged | Skipped without reading the file |
| Timestamp changed, content identical | Recorded as unchanged, nothing embedded |
| Content changed | Embedded again; unchanged parts of text files keep their vectors |
| File moved or renamed | Recognised by its content hash; only the path is updated. This also works when the file moves from one tracked location to another |
| File deleted, or now ignored | Removed from the index |
| New file | Embedded |

A move is recognised when a new file has the same content as an indexed file that is no longer
at its old path. If the old file is still there, the new one is a copy and is treated as a new
file. A file moved into a location whose rules leave it out (an ignored pattern, a file type
that location does not include) is removed from the index, as any other excluded file would be.

Files up to 64 MB are hashed completely. Larger files are identified by their size plus three
1 MB samples from the start, middle and end.

### Waiting files

Anything in a plan that you choose not to embed now is recorded as waiting. `semantix status`
shows the count and the next `semantix sync` offers it again. A changed file that is waiting
stays searchable with its old content, marked as changed since the last sync.

### Large files

Files above `limits.text_mb` (5), `limits.image_mb` (25), `limits.video_mb` (100) or
`limits.audio_mb` (100) are listed with their size, duration and estimated time, and you are
asked whether to include them. With `--yes` they are included; add `--skip-large` to leave them
out. Set a limit to 0 to turn its warning off.

### Estimates

The first sync on a machine uses built-in rates and says so. For NVIDIA GPUs those rates come
from a Tesla T4, so they are pessimistic for faster cards. After that, estimates use the
speed measured on your device in earlier runs, separately for text, images and video, and
audio.

### Stopping

Ctrl-C stops a sync after the batch in progress. Files that finished are saved; the rest are
recorded as waiting. A file is saved only when all of its parts are done, so interrupting a
long video discards the work on that one video.

### Other behaviour

- **Paths.** Every path is resolved to its full real location before it is stored, so
  `semantix sync newfile.py` from inside a tracked directory and `semantix sync
  ~/projects/newfile.py` from anywhere are the same thing.
- **Nested locations.** Syncing a directory that contains already tracked locations absorbs
  them into the new one, keeping their vectors.
- **Unavailable locations.** If a tracked directory does not exist at sync time (an unplugged
  drive, for example), it is skipped with a note and its index entries are kept. Use
  `semantix forget` if it is gone for good.
- **Capabilities turned off.** Files of a type whose capability is off are left alone: not
  embedded, and not removed if they were embedded earlier.
- **One sync at a time.** A second sync started while one is running is refused.

## How search behaves

### Modes

- **hybrid** (default): vector similarity, plus a boost for parts that contain the query's
  words.
- **vector**: similarity only.
- **keyword**: full-text match only. This works without the model.

### One query, two embeddings

The model is trained with different instructions for code search and for general search. A
query is embedded both ways; code parts are compared against the code version and everything
else against the general version.

### Scores

Raw similarity is not comparable across types: unrelated code tends to score around 0.65
against any query, while a correct image often scores lower than that. To make one ranked list
meaningful, each part's score is its similarity to your query **minus its average similarity
to 32 fixed, unrelated reference queries**. A part that is equally similar to everything ends
up near zero.

In hybrid mode up to 0.08 is added for keyword matches, in proportion to how many of the
query's words the part contains. Rare words count more, and filler words such as "the" and
"with" are ignored.

As a rough guide from testing on small collections: above 0.10 is a strong match, and below
0.05 is weak and shown dimmed. `--json` output includes both the score and the raw similarity.

`--like FILE` compares against a file instead of a text query and reports raw similarity. For
a video or audio file it uses the first segment; for a text file, the first part.

### Result labels

| Label | Meaning |
| --- | --- |
| `keyword match` | The part contains at least half of the query's meaningful words |
| `soundtrack` | The match is in a video's audio, not its picture |
| `changed since last sync` | The file was modified and is waiting to be embedded again |
| `file is gone` | The file no longer exists at that path |

## Settings

Settings live in `~/.semantix/config.toml`. Change them with `semantix config set`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `model.id` | `google/embeddinggemma-2` | Hugging Face model id |
| `model.capabilities` | `text,image,video,audio` | What is loaded into memory; text is always on |
| `model.dimension` | `768` | Stored vector size: 768, 512, 256 or 128 |
| `model.device` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `model.dtype` | `auto` | `auto`, `float32` or `bfloat16` |
| `model.batch_size` | `0` | Text batch size; 0 picks 32 on CUDA, 16 on Metal, 4 on CPU |
| `model.media_batch_size` | `0` | Image batch size; 0 picks 8 on CUDA, 4 on Metal, 1 on CPU |
| `daemon.auto_start` | `true` | Start the service when a command needs it |
| `daemon.idle_unload_minutes` | `0` | Unload the model after this many idle minutes; 0 never |
| `index.chunk_chars` | `1600` | Target size of a text part |
| `index.respect_gitignore` | `true` | Apply `.gitignore` files |
| `index.follow_symlinks` | `false` | Follow symbolic links while scanning |
| `limits.text_mb` | `5` | Warn before embedding text files above this size |
| `limits.image_mb` | `25` | Same for images |
| `limits.video_mb` | `100` | Same for videos |
| `limits.audio_mb` | `100` | Same for audio |
| `image.tokens` | `280` | Detail per image: 70, 140, 280, 560 or 1120 |
| `video.fps` | `1.0` | Frames sampled per second |
| `video.segment_seconds` | `30` | Segment length; shortened automatically if it would not fit the model |
| `video.frame_tokens` | `140` | Detail per frame: 70, 140 or 280 |
| `video.audio_track` | `true` | Also embed the soundtrack |
| `audio.segment_seconds` | `30` | Segment length, at most 300 |
| `search.limit` | `10` | Default number of results |
| `search.mode` | `hybrid` | Default mode |
| `search.per_file` | `2` | Maximum results from one file |

### Presets

`semantix config preset NAME` sets three values at once.

| Preset | `image.tokens` | `video.frame_tokens` | `video.fps` |
| --- | --- | --- | --- |
| `quality` | 280 | 140 | 1.0 |
| `balanced` | 140 | 70 | 0.5 |
| `fast` | 70 | 70 | 0.25 |

### What a change affects

- **Image, video, audio and chunk settings** apply to files embedded from then on. Existing
  vectors stay valid and searchable. Run `semantix sync --force PATH` to redo a location.
- **Lowering `model.dimension`** shortens the stored vectors in place. Nothing is embedded
  again.
- **Raising `model.dimension` or changing `model.id`** invalidates the stored vectors. You are
  asked to confirm, every file becomes waiting, and `semantix sync` embeds everything again.
- **Capabilities, device and dtype** make a running service reload the model. This is refused while a
  sync is running; restart the service afterwards to apply it.

## The background service

### Start and stop

`semantix start` returns when the model is loaded, which takes a few seconds once the files
are in the disk cache and longer on the first start after a reboot. With `daemon.auto_start`
on (the default), any command that needs the service starts it, so `start` is optional.

`semantix stop` ends the service and frees the memory.

### Autostart

```bash
semantix autostart enable
```

- **Linux:** writes `~/.config/systemd/user/semantix.service` and enables it. It starts at
  your next login. For it to run before you log in, enable lingering for your user with
  `loginctl enable-linger`.
- **macOS:** writes `~/Library/LaunchAgents/com.semantix.daemon.plist` and loads it.

`semantix autostart disable` removes it again.

### Idle unload

`semantix config set daemon.idle_unload_minutes 30` makes the service drop the model after 30
idle minutes. The service itself stays up, and the next search loads the model again, which
adds a few seconds to that one search.

### Files

| Path | Purpose |
| --- | --- |
| `~/.semantix/run/daemon.sock` | The socket, readable only by you |
| `~/.semantix/run/daemon.pid` | Process id of the service |
| `~/.semantix/logs/daemon.log` | Service log, rotated at 2 MB |

If the home path is too long for a Unix socket, the socket is placed in the system temporary
directory instead; `semantix status --json` shows where.

## Storage and privacy

```
~/.semantix/
├── config.toml      settings
├── ignore           global ignore rules
├── index/           semantix.db: files, parts, vectors and the keyword index
├── models/          the downloaded model
├── eval/            the eval set, its embeddings and reports, once `semantix eval` has run
├── run/             socket and pid of the service
└── logs/            service log
```

The index is a single SQLite database. Each part stores its vector as 4 bytes per dimension,
so a 768-dimension part costs about 3 KB plus its text. On start the service loads all vectors
into memory and searches them exactly; there is no approximate index.

The text of every text part is stored in the index, for previews and keyword search. Keep that
in mind before syncing directories with sensitive content, and note that secrets files are
ignored by the built-in rules.

The only network access is the model download during `semantix setup`, and the eval set
download the first time you run `semantix eval`. The service runs with Hugging Face's offline
mode on and binds only a local socket.

## Performance and memory

Measured on a 4-core x86 CPU with no GPU, in float32:

| Work | Speed |
| --- | --- |
| Text and code | About 3,000 characters per second (0.5 s per part) |
| Image at 280 tokens | 4.1 s |
| Image at 140 tokens | 1.8 s |
| Image at 70 tokens | 0.7 s |
| Video frame at 140 tokens | About 2 s, so a 30-second segment at 1 fps takes about a minute |
| Audio | About 0.1 s per second of audio |
| Search | About 0.1 s |
| Service start | About 8 s with warm disk cache |

| Loaded capabilities | Service memory |
| --- | --- |
| Text only | About 2.1 GB |
| Everything | About 4 GB |

A Tesla T4 measured about 4 times faster for text and 6 times faster for images, video and
audio; see the [results](#results) table.

bfloat16 uses about a third less memory on a CPU, but its speed depends on the processor: it
was twice as fast as float32 on a CPU with the AVX512-BF16 instructions and three times slower
on one without. float32 stays the CPU default. `semantix benchmark` tells you which is faster
on yours; the [results](#results) table has both cases.

While a sync is embedding, a search waits for the batch in progress. On a CPU a single video
segment can hold a search up for a while.

## Benchmark and GPU testing

```bash
semantix benchmark
```

The benchmark does four things and writes a JSON report next to where you ran it
(`semantix-benchmark-<device>.json`).

1. **Limits.** The longest video segment allowed at each detail level, and audio segments of
   30, 60 and 300 seconds, are run through the model's input processing to confirm that they
   fit and that nothing is cut off.
2. **Correctness.** A fixed set of text, image, video and audio inputs is embedded on the
   device in each number format and compared with the same inputs embedded on the CPU in
   float32. Each comparison must reach a cosine similarity of 0.99. This is what catches a
   GPU or number format that silently produces wrong vectors.
3. **Speed.** Text, single queries, images, video and audio are timed separately, with peak
   memory. A failure in one does not hide the results of the others.
4. **A real sync and search.** A generated directory of Python source files, pictures, a
   34-second video and an audio file is synced through the same engine the service uses, then searched with
   queries whose correct answer is known.

It uses a temporary index, so your own index is not touched. It takes three to four minutes
on a 4-core CPU. On a GPU it runs a larger workload.

If a number format other than the default turns out faster and correct, the benchmark prints
the command to switch to it.

### Testing on a GPU, including Google Colab

`scripts/gpu-test.sh` installs uv if needed, installs Semantix with the PyTorch build that
matches the machine, reports whether PyTorch can see the GPU, runs the benchmark and prints
the report. It works on any Linux or macOS machine.

In Colab, choose **Runtime → Change runtime type → T4 GPU**, then run this in a cell:

```bash
!curl -LsSf https://raw.githubusercontent.com/57Ajay/Semantix/main/scripts/gpu-test.sh | bash
```

On any other machine:

```bash
curl -LsSf https://raw.githubusercontent.com/57Ajay/Semantix/main/scripts/gpu-test.sh | bash
```

Arguments after `bash -s --` are passed to the benchmark, for example
`| bash -s -- --quick`. If the script reports that a GPU is present but PyTorch cannot use it,
set the backend by hand and run it again, for example with `UV_TORCH_BACKEND=cu126` set in the
environment.

### Results

All rows use the default detail settings: 280 tokens per image, 140 per video frame. The raw
reports for the first two machines are in [`benchmarks/`](benchmarks).

| Machine | Format | Text | One query | Image | Video | Audio | Peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA Tesla T4 (Colab), batches 32 and 8 | float32 | 13,100 chars/s | 60 ms | 0.55 s each | 3.9 frames/s | 78x realtime | 7.3 GB GPU |
| NVIDIA Tesla T4 (Colab), batches 32 and 8 | bfloat16 | 12,300 chars/s | 80 ms | 0.94 s each | 2.7 frames/s | 71x realtime | 7.4 GB GPU |
| Intel Xeon 3.1 GHz, 8 vCPU, 4 threads used | float32 | 3,600 chars/s | 115 ms | 3.3 s each | 0.7 frames/s | 14x realtime | 4.1 GB |
| Intel Xeon 3.1 GHz, 8 vCPU, 4 threads used | bfloat16 | 1,200 chars/s | 184 ms | 8.5 s each | 0.2 frames/s | 5x realtime | 2.6 GB |
| Intel Xeon 2.1 GHz with AVX512-BF16, 4 threads | float32 | 3,200 chars/s | 107 ms | 3.8 s each | 0.6 frames/s | 12x realtime | 4.1 GB |
| Intel Xeon 2.1 GHz with AVX512-BF16, 4 threads | bfloat16 | 6,700 chars/s | 73 ms | 1.3 s each | 1.9 frames/s | 26x realtime | 2.8 GB |
| Apple Silicon | | not measured yet | | | | | |

On the T4 the full run passed: every embedding matched the CPU reference (cosine 0.9997 or
better in both formats), the longest allowed video and audio segments fit, and a real sync of
31 files into 498 parts finished in 66 seconds with no failures and the expected file ranked
first for all six test queries.

What these show:

- **A T4 is about 4 times faster than a 4-thread CPU for text and about 6 times faster for
  images, video and audio.** That is less than a GPU usually gains. The T4 has no accelerated
  bfloat16, and float16, which it does accelerate, cannot be used with this model. Newer GPUs
  should do better, but none has been measured.
- **On the T4, float32 beats bfloat16**: 1.7 times faster for images, 1.5 times for video,
  slightly faster for text, with the same peak GPU memory. `model.dtype = auto` therefore
  picks float32 on GPUs older than the generation that accelerates bfloat16 (compute
  capability below 8.0, which includes the T4) and bfloat16 on newer ones. The newer-GPU half
  of that rule follows the model's documentation and has not been measured here.
- **Peak GPU memory was 7.3 GB** with the default batch sizes. Which stage reaches it is not
  known from this run; later reports record the peak per stage. On a GPU with less memory,
  Semantix halves the batch and retries when it runs out, and `model.batch_size` and
  `model.media_batch_size` can be lowered by hand.
- **bfloat16 on a CPU depends on the processor.** It was twice as fast on a CPU with the
  AVX512-BF16 instructions and three times slower on one without. float32 stays the CPU
  default; switch only if the benchmark on your machine says so.
- **A single query costs 60 to 190 ms everywhere.** Most of that is fixed overhead, so a GPU
  matters for indexing far more than for searching.

### A bug this benchmark found

Until it was fixed, the model's input was capped at 2,048 tokens. Text never reaches that, but
a video segment does: 30 frames at 140 tokens need about 3,700. Any video segment longer than
about 16 frames failed, on every device. It surfaced on the first T4 run, which was the first
to embed a full 30-frame segment. The cap is now the model's real limit of 8,192
tokens, segments are sized so they cannot exceed it, and the benchmark checks the longest
allowed video and audio segments on every run. The same review found that audio segments
longer than 30 seconds were being cut to 30 by the audio processor; that is fixed too. The
default 30-second audio segments were never affected.

## Measuring search quality

```bash
semantix eval
```

The benchmark checks that the model runs correctly and how fast. `semantix eval` checks
whether search finds the right things. It runs a fixed collection of files and queries whose
correct answers are known, and reports how often and how high the right answer is ranked.
Use it before and after any change to chunking, scoring, settings or the model, and when
comparing devices or number formats.

The first run downloads the standard eval set (about 17 MB) from the `eval` branch of the
repository into `~/.semantix/eval/set`. The set lives on its own branch, so cloning or
installing Semantix never downloads it. `--update` fetches the newest version. The collection is embedded into a separate index under
`~/.semantix/eval/work`; your own index is never touched. On a 4-core CPU the first run takes
about 18 minutes, more than half of it video. Later runs reuse every embedding, including the
query embeddings, so changing a search-time option such as `--keyword-bonus` reruns in seconds
without loading the model. A setting that changes what gets embedded starts from the closest
earlier run and embeds again only the affected types: `index.chunk_chars` re-embeds text and
code, `image.tokens` only images.

Every run is saved as a JSON report in `~/.semantix/eval/reports`.

### The standard set

| Part | Contents |
| --- | --- |
| Code | boltons (17 Python modules), express (JavaScript) and cobra (Go), with their docs |
| Notes | 36 short personal notes: work, home, recipes, travel, learning, a journal |
| Images | Photos (CC0 or public domain), NASA pictures, and drawn screenshots, a receipt, a slide, a diagram, a boarding pass |
| Video | A shuttle launch with commentary, a 1951 public-domain film, an ISS pass over a hurricane, a 1906 street film |
| Audio | 13 National Park Service sound recordings, three audiobook excerpts, two music pieces |

It deliberately contains near misses: two budgets, a cat in the snow next to a dog in the
snow, rain next to a stream and the ocean, a photo of a hurricane next to a video of one.
Without them every method scores close to 100% and the set cannot tell them apart.

The queries are grouped by what they test:

| Category | What it tests |
| --- | --- |
| code-intent | A plain-language request finds the function that does it |
| code-identifier | Exact names and error messages |
| docs | Prose documentation inside the projects |
| notes | Notes found by their topic |
| paraphrase | The note says the same thing in other words |
| image-content | What a photo shows |
| image-text | Text inside a picture: receipt, slide, screenshot, chart |
| video-moment | The right video and the right segment |
| speech | What is said in audiobooks and video soundtracks |
| audio | Sounds and music |
| cross-type | The answer exists as two types, for example a note and a screenshot |
| like | `--like` finds similar files |
| no-answer | Nothing in the collection fits |

Queries are split into `dev` and `test`. Tune with `--split dev` and look at `test` only
before keeping a change, so the set does not end up tuned to itself.

### How results are scored

A result matches an answer when it is the same file and, if the answer gives a line range or
a time range, the result's part overlaps it. An answer is graded `relevant` (the answer) or
`acceptable` (also fine, counts half and is never a miss); `not_relevant` records that a
result was checked and is wrong.

| Column | Meaning |
| --- | --- |
| Top 1 | Share of queries whose first result is a relevant answer |
| Recall@5, Recall@10 | Share of the relevant answers found in the first 5 or 10 results |
| MRR | Mean of 1 / rank of the first relevant answer, 0 if it is not in the first 10 |
| nDCG@10 | Ranking quality of the first 10, with acceptable answers counting half |

Below each table:

- **Right segment or lines**: for queries whose answer has a range, how often the first result
  from the right file is the right part of it.
- **No-answer queries**: the score threshold that best separates queries that have an answer
  from queries that do not, and how many it gets right. This shows whether `--min-score` can
  hide nonsense results.
- **Searching all types**: the same queries run once over everything and once restricted to
  the answer's own type. The gap shows how much the score calibration between types costs.
- **Not in the top 5**: each query that missed, with what was found instead.

### Comparing changes

```bash
semantix eval --split dev
```

```bash
semantix eval --split dev --compare last --config index.chunk_chars=1200
```

`--compare` prints the change per category and lists the queries that moved, with their old
and new rank. It also notes any difference in settings or set version between the two runs.

### Results

Standard set version 2 on an Intel Xeon 2.1 GHz with 4 threads, float32, default settings.
The full report is in `results/` on the `eval` branch.

| Category | Queries | Top 1 | Recall@5 | MRR |
| --- | --- | --- | --- | --- |
| code-intent | 20 | 55% | 90% | 67% |
| code-identifier | 6 | 67% | 100% | 83% |
| docs | 6 | 67% | 100% | 81% |
| notes | 12 | 92% | 100% | 96% |
| paraphrase | 8 | 88% | 100% | 94% |
| image-content | 23 | 96% | 100% | 98% |
| image-text | 7 | 71% | 100% | 86% |
| video-moment | 7 | 100% | 100% | 100% |
| speech | 6 | 100% | 100% | 100% |
| audio | 14 | 29% | 57% | 42% |
| cross-type | 6 | 100% | 100% | 100% |
| like | 4 | 75% | 75% | 79% |
| **all, hybrid** | 119 | **76%** | **92%** | **83%** |
| all, vector only | 119 | 75% | 88% | 81% |
| all, keyword only | 115 | 37% | 45% | 41% |

The first run embedded 158 files into 860 parts in about 18 minutes. Every later run of the
same settings takes about 15 seconds.

What these show:

- **Images, video moments and speech are close to perfect.** A photo is found from a plain
  description, the right 30-second video segment is found for a scene, and recordings are
  found by what is said in them, including the commentary and narration in video soundtracks.
- **Sounds are the weak spot.** Short field recordings are confused with each other: a wolf
  howl ranks below a siren, rain below thunder and a stream. Writing "the sound of rain"
  instead of "rain falling" made no difference (MRR 43% and 42%). When all types are searched,
  8 of the 14 sound queries rank lower than with `--type audio`, because photos of the same
  scene compete. Use `--type audio` when looking for a sound.
- **Code is found, but often not first.** The right function is in the top 5 for 90% of the
  plain-language code queries, first for 55%. Most misses rank a neighbouring function or the
  project's documentation about the same feature first, which is often a reasonable answer.
- **Hybrid beats vector-only mostly on code**: MRR 67% against 59% for code-intent and 83%
  against 66% for exact names. Keyword-only search cannot find media at all.
- **A low top score really does mean "probably nothing here".** The best result of the eight
  queries with no answer scored between 0.02 and 0.12; every correct first result scored at
  least 0.13. That gap is from only eight queries, so treat 0.12 as a rough guide for
  `--min-score`, not a rule.

### What the first experiments showed

| Change | Overall MRR | Recall@5 | Notes |
| --- | --- | --- | --- |
| Default (calibrated, keyword bonus 0.08) | 83.0% | 92.4% | |
| Without score calibration | 81.1% | 86.1% | Exact names 100% instead of 83%, paraphrases 75% instead of 94% |
| Keyword bonus 0 | 81.0% | 88.2% | The same as vector only |
| Keyword bonus 0.04 | 82.7% | 91.6% | |
| Keyword bonus 0.12 | 83.3% | 92.4% | One query better than the default |
| Keyword bonus 0.24 | 82.5% | 91.2% | |
| `index.chunk_chars` 800 | 82.4% | 89.9% | 1,533 parts instead of 860; docs better, code and `--like` worse |
| `index.chunk_chars` 2400 | 83.9% | 92.4% | 640 parts; 8 of the 9 queries that moved went up, on both splits |

- **Calibration is worth keeping, but it costs exact matches.** Subtracting each part's
  similarity to unrelated queries helps prose a lot and hurts code identifiers. Calibrating
  per type, or weighting keyword matches more when the query looks like an identifier, are
  the obvious next experiments.
- **The keyword bonus is flat between 0.08 and 0.16.** The default stays.
- **Larger text parts did slightly better, smaller ones slightly worse.** At 2,400 characters
  MRR rose by about one query's worth on both the dev and the test split, with a quarter
  fewer parts to store and search. The gain is too small to change the default on this
  evidence alone; sizes between 2,000 and 3,200, ideally with structure-aware chunking for
  code, are worth measuring next. Each of these runs re-embedded only text and code and took
  about 4 to 5 minutes.

### Writing your own set

An eval set is a folder:

```
my-set/
├── set.toml        name = "my-set", version = "1"
├── queries.toml
└── corpus/         the files to search
```

```toml
[[query]]
id = "retry-backoff"
text = "retry a request with exponential backoff"
category = "code"
split = "dev"
relevant = [{ path = "src/http.py", lines = [120, 160] }]
acceptable = ["docs/networking.md"]
not_relevant = ["src/queue.py"]

[[query]]
id = "launch-crowd"
text = "people watching the launch"
category = "video"
relevant = [{ path = "videos/launch.mp4", seconds = [15, 25] }]

[[query]]
id = "similar-to-curry"
like = "notes/curry.md"
category = "like"
relevant = ["notes/soup.md"]
```

Paths are relative to `corpus/`. A query without any `relevant` answer is a no-answer query.
Run it with `semantix eval my-set`. After the first run, `--unjudged` lists the top results
that have no grade yet; grading them prevents a correct result from counting as a miss just
because nobody graded it.

## Architecture

| Module | Role |
| --- | --- |
| `cli.py` | Commands, prompts and output |
| `client.py` | Talks to the service; starts and stops it |
| `server.py` | The service: socket protocol, model loading, idle unload |
| `engine.py` | Planning and running syncs, search, forget, check |
| `scanner.py` | Walking directories, applying selection, hashing files |
| `ignore.py` | Layered ignore rules |
| `chunking.py` | Splitting text into parts |
| `media.py` | Decoding images, video and audio |
| `model.py` | Loading the model and embedding each modality |
| `store.py` | The SQLite database |
| `vindex.py` | The in-memory vector index and score calibration |
| `config.py`, `paths.py` | Settings and file locations |
| `autostart.py` | systemd and launchd integration |
| `benchmark.py` | The benchmark: correctness check, timings, real sync and search |
| `evaluate.py` | The eval command: eval sets, scoring, reports and comparisons |

The CLI and the service exchange one JSON object per line. A request is `{"op": ..., "args":
...}`; the service answers with zero or more progress events and a final `{"ok": true,
"result": ...}` or `{"ok": false, "error": ...}`. Closing the connection during a sync cancels
it.

## Development

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check src tests
```

The tests use a small fake embedder, so they run in about a second and do not need the model.
That also means they cannot tell whether search got better or worse. `semantix eval` is the
acceptance test for any change to chunking, scoring or settings, and `semantix benchmark` for
any change to the model path.

## Limits and untested areas

- **CUDA is verified on one GPU, Apple Metal on none.** The full benchmark passed on a Tesla
  T4 in both number formats. No other NVIDIA GPU has been run, so the automatic choice of
  bfloat16 on newer GPUs is untested, and the background service itself was not run on a GPU
  (the benchmark uses the same engine without the service around it). Apple Metal has never
  been run. Run `semantix benchmark` on a new machine to check it.
- **Autostart has not been exercised end to end.** The systemd unit and launchd plist are
  generated and checked by tests, but enabling them was not possible in the development
  environment. macOS as a whole is untested.
- **Windows is not supported.** The service uses Unix sockets.
- **PDFs and office documents are not embedded.**
- **Vectors are searched exactly, in memory.** That is fast up to a few hundred thousand parts
  (1 GB of memory per 350,000 parts at 768 dimensions) and has not been tested beyond small
  collections.
- **Score thresholds are rough.** On the standard eval set the best result of a query with no
  answer never scored above 0.12, and every correct first result scored at least 0.13, but
  that comes from only eight such queries. See [measuring search quality](#measuring-search-quality).
- **A long video is saved only when it completes**, so interrupting one loses that file's
  progress.
