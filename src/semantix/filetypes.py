from __future__ import annotations

import os

KINDS = ("code", "text", "image", "video", "audio")
TEXT_KINDS = ("code", "text")
KIND_CAPABILITY = {
    "code": "text",
    "text": "text",
    "image": "image",
    "video": "video",
    "audio": "audio",
}

CODE_EXTENSIONS = frozenset(
    """
    .py .pyi .ipynb .js .jsx .mjs .cjs .ts .tsx .vue .svelte .astro .java .kt .kts .scala .groovy
    .gradle .c .h .cc .cpp .cxx .hpp .hh .cs .fs .vb .go .rs .rb .erb .php .swift .m .mm .dart .lua
    .pl .pm .r .jl .ex .exs .erl .hs .ml .clj .cljs .lisp .el .zig .nim .v .sv .vhd .sol .asm .s
    .sh .bash .zsh .fish .ps1 .bat .cmd .sql .graphql .gql .proto .thrift .html .htm .xml .css .scss
    .sass .less .json .jsonc .json5 .yaml .yml .toml .ini .cfg .conf .properties .tf
    .tfvars .hcl .nix .dockerfile .mk .cmake .bzl .tpl .j2 .jinja .hbs .ejs .pug .twig .vim
    """.split()
)

TEXT_EXTENSIONS = frozenset(
    ".md .mdx .markdown .txt .text .rst .adoc .org .tex .bib .csv .tsv .log .srt .vtt".split()
)

IMAGE_EXTENSIONS = frozenset(".png .jpg .jpeg .webp .gif .bmp .tif .tiff".split())
VIDEO_EXTENSIONS = frozenset(".mp4 .mov .mkv .webm .avi .m4v .wmv .flv .mpg .mpeg".split())
AUDIO_EXTENSIONS = frozenset(".mp3 .wav .flac .ogg .oga .opus .m4a .aac .wma".split())

CODE_FILENAMES = frozenset(
    """
    dockerfile containerfile makefile gnumakefile cmakelists.txt jenkinsfile vagrantfile procfile
    gemfile rakefile brewfile justfile caddyfile tiltfile build workspace
    .gitignore .gitattributes .dockerignore .editorconfig .npmrc .babelrc .eslintrc .prettierrc
    .bashrc .zshrc .profile .bash_profile .vimrc .tmux.conf
    """.split()
)

TEXT_FILENAMES = frozenset("readme license licence changelog authors contributing notes todo".split())

SNIFF_BYTES = 4096


def extension_of(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def looks_like_text(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            sample = handle.read(SNIFF_BYTES)
    except OSError:
        return False
    if not sample:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.start < len(sample) - 4:
            return False
    return True


def classify(path: str, sniff: bool = True) -> str | None:
    name = os.path.basename(path).lower()
    extension = extension_of(name)
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in CODE_EXTENSIONS or name in CODE_FILENAMES:
        return "code"
    if extension in TEXT_EXTENSIONS or name in TEXT_FILENAMES:
        return "text"
    if not sniff or not looks_like_text(path):
        return None
    try:
        with open(path, "rb") as handle:
            if handle.read(2) == b"#!":
                return "code"
    except OSError:
        return None
    return "text"
