from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from . import __version__, client
from .config import CAPABILITIES, DIMENSIONS, OPTIONS, PRESETS, Config, ConfigError
from .filetypes import KINDS
from .ignore import GLOBAL_IGNORE_TEMPLATE
from .paths import Paths, display_path, get_paths

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    help="Local semantic search over your code, notes, images, video and audio.",
)
config_app = typer.Typer(help="Show or change settings.", invoke_without_command=True, rich_markup_mode="rich")
autostart_app = typer.Typer(help="Start the service automatically at login.", no_args_is_help=True)
app.add_typer(config_app, name="config")
app.add_typer(autostart_app, name="autostart")

console = Console(highlight=False)
errors = Console(stderr=True, highlight=False)

WEAK_SCORE = 0.05
KIND_STYLE = {"code": "cyan", "text": "green", "image": "magenta", "video": "yellow", "audio": "blue"}


def fail(message: str) -> typer.Exit:
    errors.print(f"[bold red]error:[/] {message}")
    return typer.Exit(1)


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 1:
        return "under 1s"
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def estimate(seconds: float) -> str:
    return "under 1s" if seconds < 1 else f"about {human_time(seconds)}"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def ago(moment: float | None) -> str:
    if not moment:
        return "never"
    delta = time.time() - moment
    if delta < 90:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


def resolve(path: Path | str) -> str:
    return os.path.realpath(os.path.expanduser(str(path)))


def split_values(values: list[str] | None) -> list[str]:
    result = []
    for value in values or []:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def model_present(paths: Paths, config: Config) -> bool:
    folder = paths.models / ("models--" + config.get("model.id").replace("/", "--")) / "snapshots"
    return folder.is_dir() and any(folder.iterdir())


def ensure_service(paths: Paths, config: Config) -> None:
    state = client.ping(paths)
    if state is not None and state["model_state"] in ("ready", "unloaded"):
        return
    if state is None and not config.get("daemon.auto_start"):
        raise fail("the semantix service is not running, start it with `semantix start`")
    start_service(paths, config)


def start_service(paths: Paths, config: Config) -> dict[str, Any]:
    if client.ping(paths) is None and not model_present(paths, config):
        raise fail("the model is not downloaded yet, run `semantix setup` first")
    try:
        with errors.status("Starting semantix…") as status:
            return client.start(paths, on_wait=lambda stage: status.update(f"Starting semantix… {stage}"))
    except (client.DaemonUnavailable, client.RemoteError) as error:
        raise fail(str(error)) from None


def remote(paths: Paths, op: str, args: dict[str, Any] | None = None) -> Any:
    try:
        return client.request(paths, op, args)
    except (client.DaemonUnavailable, client.RemoteError) as error:
        raise fail(str(error)) from None


def with_engine(paths: Paths, op: str, args: dict[str, Any]) -> Any:
    if client.ping(paths) is not None:
        return remote(paths, op, args)
    from .engine import Engine, EngineError

    engine = Engine(paths)
    try:
        return getattr(engine, op)(**args)
    except EngineError as error:
        raise fail(str(error)) from None
    finally:
        engine.close()


def version_callback(value: bool) -> None:
    if value:
        console.print(f"semantix {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=version_callback, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    pass


@app.command()
def setup(
    capabilities: Annotated[str | None, typer.Option(help="Comma separated: text,image,video,audio")] = None,
    dimension: Annotated[int | None, typer.Option(help="Vector size: 768, 512, 256 or 128")] = None,
    preset: Annotated[str | None, typer.Option(help="quality, balanced or fast")] = None,
    autostart: Annotated[bool | None, typer.Option("--autostart/--no-autostart", help="Start at login")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Accept defaults without asking")] = False,
) -> None:
    """Choose capabilities, download the model and start the service."""
    paths = get_paths().ensure()
    config = Config.load(paths.config)
    ask = interactive() and not yes
    console.print(f"[bold]Semantix setup[/]  [dim]{paths.home}[/]")

    with console.status("Checking hardware…"):
        from .model import resolve_device

        try:
            device = resolve_device(config.get("model.device"))
        except Exception:
            device = "cpu"
    label = {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (Metal)", "cpu": "CPU"}[device]
    console.print(f"Device: [bold]{label}[/]")

    try:
        if capabilities is not None:
            config.set("model.capabilities", capabilities)
        elif ask:
            import questionary

            current = config.get("model.capabilities")
            sizes = {"image": "images (+170M parameters)", "video": "video (uses the image encoder)", "audio": "audio (+300M parameters)"}
            picked = questionary.checkbox(
                "Text and code are always on. What else should be searchable?",
                choices=[questionary.Choice(sizes[name], value=name, checked=name in current) for name in CAPABILITIES[1:]],
            ).ask()
            if picked is None:
                raise typer.Exit(1)
            config.set("model.capabilities", ["text", *picked])

        if preset is None and ask and device == "cpu" and (config.has("image") or config.has("video")):
            import questionary

            preset = questionary.select(
                "No GPU found. Images and video are slow on a CPU, pick a trade-off:",
                choices=[
                    questionary.Choice("balanced: half the detail, about 2x faster (recommended)", value="balanced"),
                    questionary.Choice("quality: the model's full detail, slowest", value="quality"),
                    questionary.Choice("fast: least detail, about 5x faster", value="fast"),
                ],
            ).ask()
            if preset is None:
                raise typer.Exit(1)
        if preset is not None:
            if preset not in PRESETS:
                raise ConfigError(f"unknown preset {preset!r}, choose from {', '.join(PRESETS)}")
            for key, value in PRESETS[preset].items():
                config.set(key, value)

        if dimension is not None:
            config.set("model.dimension", dimension)
        elif ask:
            import questionary

            notes = {768: "best quality (recommended)", 512: "1.5x smaller", 256: "3x smaller, slight quality loss", 128: "6x smaller, text only use"}
            chosen = questionary.select(
                "Vector size to store:",
                choices=[questionary.Choice(f"{size}: {notes[size]}", value=size) for size in DIMENSIONS],
                default=None,
            ).ask()
            if chosen is None:
                raise typer.Exit(1)
            config.set("model.dimension", chosen)
    except ConfigError as error:
        raise fail(str(error)) from None

    config.save(paths.config)
    if not paths.ignore.exists():
        paths.ignore.write_text(GLOBAL_IGNORE_TEMPLATE)

    if model_present(paths, config):
        console.print("Model: already downloaded")
    else:
        console.print(f"Downloading [bold]{config.get('model.id')}[/] (about 1.5 GB)…")
        from .model import download

        try:
            download(config.get("model.id"), paths.models)
        except Exception as error:
            raise fail(f"download failed: {error}") from None
        console.print("Model: downloaded")

    if autostart is None and ask:
        import questionary

        autostart = questionary.confirm("Start semantix automatically when you log in?", default=False).ask()
    if autostart:
        from . import autostart as autostart_module

        try:
            location = autostart_module.enable(paths.log)
            console.print(f"Autostart: enabled ({display_path(str(location))})")
        except autostart_module.AutostartError as error:
            errors.print(f"[yellow]Autostart was not enabled:[/] {error}")

    if client.ping(paths) is not None:
        client.stop(paths)
    state = start_service(paths, config)
    console.print(f"Service: running (pid {state['pid']})")
    console.print("\nNext: [bold]semantix sync <directory>[/] then [bold]semantix search \"…\"[/]")


@app.command()
def start(
    foreground: Annotated[bool, typer.Option("--foreground", help="Run in this terminal instead of the background")] = False,
) -> None:
    """Start the background service and load the model."""
    paths = get_paths().ensure()
    config = Config.load(paths.config)
    if foreground:
        from .engine import EngineError
        from .server import AlreadyRunning, run

        try:
            run(paths)
        except AlreadyRunning:
            console.print("semantix is already running")
        except EngineError as error:
            raise fail(str(error)) from None
        return
    state = client.ping(paths)
    if state is not None and state["model_state"] in ("ready", "unloaded"):
        console.print(f"semantix is already running (pid {state['pid']})")
        return
    state = start_service(paths, config)
    console.print(f"semantix is running (pid {state['pid']})")


@app.command()
def stop() -> None:
    """Stop the background service and free its memory."""
    paths = get_paths()
    if client.stop(paths):
        console.print("semantix stopped")
    else:
        console.print("semantix is not running")


@app.command()
def restart() -> None:
    """Restart the background service."""
    paths = get_paths().ensure()
    client.stop(paths)
    state = start_service(paths, Config.load(paths.config))
    console.print(f"semantix is running (pid {state['pid']})")


def offline_status(paths: Paths, config: Config) -> dict[str, Any]:
    from .store import Store

    index: dict[str, Any] = {"files": {}, "chunks": {}, "vectors": 0, "db_bytes": 0}
    roots: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if paths.db.exists():
        store = Store(paths.db, readonly=True)
        try:
            index.update(store.counts())
            index["vectors"] = sum(index["chunks"].values())
            index["db_bytes"] = paths.index_bytes()
            roots = store.root_summaries()
            failures = store.error_files()
        finally:
            store.close()
    return {
        "pid": None,
        "home": str(paths.home),
        "model": {
            "id": config.get("model.id"), "state": "stopped", "device": None, "dtype": None,
            "capabilities": config.get("model.capabilities"), "dimension": config.get("model.dimension"),
            "downloaded": model_present(paths, config),
        },
        "index": index,
        "roots": roots,
        "errors": failures,
        "job": None,
    }


@app.command()
def status(as_json: Annotated[bool, typer.Option("--json", help="Print machine readable output")] = False) -> None:
    """Show the service, model and index state."""
    paths = get_paths()
    config = Config.load(paths.config)
    running = client.ping(paths) is not None
    info = remote(paths, "status") if running else offline_status(paths, config)
    if as_json:
        console.print_json(json.dumps(info))
        return
    from . import autostart as autostart_module

    model = info["model"]
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")
    if running:
        table.add_row("Service", f"[green]running[/] (pid {info['pid']}, up {human_time(info['uptime'])})")
    else:
        table.add_row("Service", "[yellow]stopped[/]  start it with `semantix start`")
    state = model["state"]
    if state == "ready":
        detail = f"[green]loaded[/] on {model['device']} ({model['dtype']})"
    elif state == "loading":
        detail = "[yellow]loading…[/]"
    elif state == "unloaded":
        detail = "[yellow]unloaded while idle[/], loads again on the next query"
    elif state == "error":
        detail = f"[red]failed:[/] {model.get('error')}"
    else:
        detail = "downloaded" if model.get("downloaded") else "[red]not downloaded[/], run `semantix setup`"
    table.add_row("Model", f"{model['id']}  {detail}")
    table.add_row("Capabilities", f"{', '.join(model['capabilities'])}  ·  {model['dimension']} dimensions")
    try:
        enabled = autostart_module.is_enabled()
        table.add_row("Autostart", "enabled" if enabled else "disabled")
    except autostart_module.AutostartError:
        pass
    table.add_row("Home", f"{display_path(info['home'])}  ·  index {human_bytes(info['index']['db_bytes'])}")
    console.print(table)

    files = info["index"]["files"]
    if files:
        counts = Table(box=None, padding=(0, 2), header_style="dim")
        for column in ("Type", "Files", "Parts", "Size", "Waiting", "Errors"):
            counts.add_column(column, justify="left" if column == "Type" else "right")
        for kind in KINDS:
            entry = files.get(kind)
            if not entry:
                continue
            parts = sum(count for part, count in info["index"]["chunks"].items() if part == kind)
            counts.add_row(
                f"[{KIND_STYLE[kind]}]{kind}[/]", str(entry["files"]), str(parts), human_bytes(entry["bytes"]),
                str(entry["pending"] + entry["stale"]), str(entry["error"]),
            )
        console.print()
        console.print(counts)
        waiting = sum(entry["pending"] + entry["stale"] for entry in files.values())
        console.print(f"\n{len(info['roots'])} tracked location(s), {info['index']['vectors']} searchable parts")
        if waiting:
            console.print(f"[yellow]{waiting} file(s) are waiting to be embedded.[/] Run `semantix sync`.")
    else:
        console.print("\nNothing indexed yet. Run `semantix sync <directory>`.")
    if info["errors"]:
        console.print("\n[red]Files that failed:[/]")
        for entry in info["errors"]:
            console.print(f"  {display_path(entry['path'])}  [dim]{entry['error']}[/]")
    if info.get("job"):
        job = info["job"]
        console.print(
            f"\nSync in progress: {job['files_done']}/{job['files_total']} files, about {human_time(job['eta'])} left"
        )


@autostart_app.command("enable")
def autostart_enable() -> None:
    """Start the service at every login."""
    from . import autostart as autostart_module

    paths = get_paths().ensure()
    try:
        location = autostart_module.enable(paths.log)
    except autostart_module.AutostartError as error:
        raise fail(str(error)) from None
    console.print(f"Autostart enabled: {display_path(str(location))}")


@autostart_app.command("disable")
def autostart_disable() -> None:
    """Stop starting the service at login."""
    from . import autostart as autostart_module

    try:
        location = autostart_module.disable()
    except autostart_module.AutostartError as error:
        raise fail(str(error)) from None
    console.print("Autostart disabled" if location else "Autostart was not enabled")


@autostart_app.command("status")
def autostart_status() -> None:
    """Show whether autostart is enabled."""
    from . import autostart as autostart_module

    try:
        enabled = autostart_module.is_enabled()
        location = autostart_module.target_path()
    except autostart_module.AutostartError as error:
        raise fail(str(error)) from None
    console.print(f"Autostart is {'enabled' if enabled else 'disabled'}  [dim]{display_path(str(location))}[/]")


def render_plan(plan: dict[str, Any]) -> None:
    for note in plan["notes"]:
        console.print(f"[yellow]{note}[/]")
    for root in plan["roots"]:
        label = "  [green](new)[/]" if root["new"] else ""
        console.print(f"\n[bold]{display_path(root['path'])}[/]{label}")
        if root["groups"]:
            table = Table(box=None, padding=(0, 2), header_style="dim")
            for column in ("Type", "Files", "Size", "New", "Changed", "Est. time"):
                table.add_column(column, justify="left" if column == "Type" else "right")
            for group in root["groups"]:
                table.add_row(
                    f"[{KIND_STYLE[group['kind']]}]{group['kind']}[/]", str(group["files"]), human_bytes(group["bytes"]),
                    str(group["new"]), str(group["changed"] + group["retry"]), human_time(group["cost"]),
                )
            console.print(table)
        extras = []
        if root["deletes"]:
            extras.append(f"{root['deletes']} removed from disk")
        if root["moves"]:
            extras.append(f"{root['moves']} moved or renamed")
        extras.append(f"{root['unchanged']} unchanged")
        console.print(f"  [dim]{' · '.join(extras)}[/]")
    if plan["skipped"]:
        parts = ", ".join(f"{count} {reason}" for reason, count in sorted(plan["skipped"].items()))
        console.print(f"\n[dim]Skipped: {parts}[/]")
    for kind, count in plan["off_kinds"].items():
        console.print(f"[dim]{count} {kind} file(s) left out because that capability is turned off[/]")
    if plan["large"]:
        console.print("\n[yellow]Large files, these can take a while:[/]")
        for entry in plan["large"][:15]:
            length = f", {clock(entry['duration'])} long" if entry.get("duration") else ""
            console.print(
                f"  {display_path(entry['path'])}  [dim]{entry['kind']}, {human_bytes(entry['size'])}{length}, "
                f"{estimate(entry['cost'])}[/]"
            )
        if len(plan["large"]) > 15:
            console.print(f"  [dim]…and {len(plan['large']) - 15} more[/]")
    totals = plan["totals"]
    if totals["files"]:
        basis = "" if plan["measured"] else "  [dim](rough estimate until the first sync has run)[/]"
        console.print(
            f"\n[bold]{totals['files']} file(s), {human_bytes(totals['bytes'])}, {estimate(totals['cost'])} "
            f"on {plan['device']}[/]{basis}"
        )


def run_execute(paths: Paths, plan_id: str, groups: list[list[str]] | None, skip: list[str]) -> dict[str, Any] | None:
    progress = Progress(
        SpinnerColumn(),
        BarColumn(bar_width=28),
        TextColumn("{task.fields[count]}"),
        TextColumn("[dim]{task.fields[eta]}[/]"),
        TextColumn("{task.description}"),
        console=console,
        transient=True,
    )
    result = None
    failures: list[tuple[str, str]] = []
    try:
        with progress:
            task = progress.add_task("", total=1.0, count="", eta="")
            for event in client.stream(paths, "execute", {"plan_id": plan_id, "groups": groups, "skip": skip}):
                kind = event["event"]
                if kind == "start":
                    progress.update(task, total=max(event["total"], 1e-9), count=f"0/{event['files_total']}")
                elif kind == "progress":
                    name = display_path(event["path"]) if event.get("path") else ""
                    if len(name) > 48:
                        name = "…" + name[-47:]
                    progress.update(
                        task, completed=min(event["done"], event["total"]), description=name,
                        count=f"{event['files_done']}/{event['files_total']}", eta=f"{human_time(event['eta'])} left",
                    )
                elif kind == "file" and event["status"] == "error":
                    failures.append((event["path"], event["error"]))
                elif kind == "result":
                    result = event["result"]
    except KeyboardInterrupt:
        console.print("\nStopped. Finished files are saved, run `semantix sync` to continue with the rest.")
        raise typer.Exit(130) from None
    except (client.DaemonUnavailable, client.RemoteError) as error:
        raise fail(str(error)) from None
    for path, message in failures[:10]:
        errors.print(f"[red]failed:[/] {display_path(path)}  [dim]{message}[/]")
    if len(failures) > 10:
        errors.print(f"[red]…and {len(failures) - 10} more, see `semantix status`[/]")
    return result


@app.command()
def sync(
    targets: Annotated[list[Path] | None, typer.Argument(help="Directories or files. Leave empty to check everything tracked.")] = None,
    only: Annotated[list[str] | None, typer.Option("--only", help="Only these types: code,text,image,video,audio")] = None,
    skip_ext: Annotated[list[str] | None, typer.Option("--skip-ext", help="Extensions to leave out, e.g. .csv,.log")] = None,
    exclude: Annotated[list[str] | None, typer.Option("--exclude", help="Ignore pattern to remember for this directory")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Embed everything in the plan without asking")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show the plan and stop")] = False,
    force: Annotated[bool, typer.Option("--force", help="Embed again even if nothing changed")] = False,
    skip_large: Annotated[bool, typer.Option("--skip-large", help="Leave out files above the size limits")] = False,
) -> None:
    """Embed new and changed files. Without arguments, checks every tracked location."""
    paths = get_paths().ensure()
    config = Config.load(paths.config)
    resolved = [resolve(target) for target in targets] if targets else None
    selection: dict[str, Any] = {}
    kinds = split_values(only)
    if kinds:
        unknown = [kind for kind in kinds if kind not in KINDS]
        if unknown:
            raise fail(f"unknown type {unknown[0]!r}, choose from {', '.join(KINDS)}")
        selection["kinds"] = kinds
    if skip_ext:
        selection["skip_ext"] = split_values(skip_ext)
    if exclude:
        selection["exclude"] = list(exclude)

    ensure_service(paths, config)
    arguments = {"targets": resolved, "selection": selection or None, "force": force}
    with console.status("Checking files…"):
        plan = remote(paths, "plan", arguments)
    render_plan(plan)
    totals = plan["totals"]
    housekeeping = totals["deletes"] or totals["moves"]
    if dry_run:
        return
    if not totals["files"]:
        if housekeeping:
            run_execute(paths, plan["id"], None, [])
            console.print("Index updated.")
        elif plan["roots"]:
            console.print("\nEverything is up to date.")
        return

    ask = interactive() and not yes
    if not ask and not yes:
        console.print("\nRun again with [bold]--yes[/] to embed these files.")
        raise typer.Exit(1)

    if ask and not selection:
        fresh = [root for root in plan["roots"] if root["new"] and root.get("inventory")]
        if fresh:
            chosen = ask_selection(fresh)
            if chosen is None:
                console.print("Nothing was changed.")
                return
            if chosen:
                arguments["selection"] = chosen
                with console.status("Checking files…"):
                    plan = remote(paths, "plan", arguments)
                render_plan(plan)
                totals = plan["totals"]
                if not totals["files"]:
                    console.print("\nNothing left to embed with that selection.")
                    return

    skip: list[str] = []
    if plan["large"]:
        if skip_large:
            skip = [entry["path"] for entry in plan["large"]]
        elif ask:
            import questionary

            include = questionary.confirm(f"Include the {len(plan['large'])} large file(s) listed above?", default=False).ask()
            if include is None:
                raise typer.Exit(1)
            if not include:
                skip = [entry["path"] for entry in plan["large"]]

    groups: list[list[str]] | None = None
    if ask:
        import questionary

        choice = questionary.select(
            "Embed these files now?",
            choices=[
                questionary.Choice("Yes, embed everything listed", value="all"),
                questionary.Choice("Let me choose what to embed now", value="choose"),
                questionary.Choice("Not now", value="later"),
            ],
        ).ask()
        if choice is None:
            raise typer.Exit(1)
        if choice == "later":
            if any(not root["new"] for root in plan["roots"]):
                run_execute(paths, plan["id"], [], [])
                console.print("Nothing embedded. The files are marked as waiting, see `semantix status`.")
            else:
                console.print("Nothing was changed.")
            return
        if choice == "choose":
            options = []
            for root in plan["roots"]:
                for group in root["groups"]:
                    label = f"{display_path(root['path'])}  {group['kind']}  {group['files']} file(s), {estimate(group['cost'])}"
                    options.append(questionary.Choice(label, value=[root["path"], group["kind"]], checked=True))
            picked = questionary.checkbox("Select what to embed now (the rest stays waiting):", choices=options).ask()
            if picked is None:
                raise typer.Exit(1)
            groups = picked

    result = run_execute(paths, plan["id"], groups, skip)
    if result is None:
        return
    summary = [f"embedded {result['embedded']} file(s) into {result['chunks']} part(s) in {human_time(result['seconds'])}"]
    if result["deleted"]:
        summary.append(f"removed {result['deleted']}")
    if result["moved"]:
        summary.append(f"moved {result['moved']}")
    if result["deferred"]:
        summary.append(f"{result['deferred']} waiting")
    if result["failed"]:
        summary.append(f"[red]{result['failed']} failed[/]")
    console.print("Done: " + ", ".join(summary))


def ask_selection(roots: list[dict[str, Any]]) -> dict[str, Any] | None:
    import questionary

    kinds: dict[str, dict[str, int]] = {}
    extensions: dict[str, dict[str, Any]] = {}
    for root in roots:
        for kind, entry in root["inventory"]["kinds"].items():
            total = kinds.setdefault(kind, {"files": 0, "bytes": 0})
            total["files"] += entry["files"]
            total["bytes"] += entry["bytes"]
        for entry in root["inventory"]["extensions"]:
            total = extensions.setdefault(entry["ext"], {"ext": entry["ext"], "files": 0, "bytes": 0})
            total["files"] += entry["files"]
            total["bytes"] += entry["bytes"]
    choice = questionary.select(
        "This location is new. What should be embedded from it?",
        choices=[
            questionary.Choice("Everything listed above", value="all"),
            questionary.Choice("Choose file types", value="kinds"),
            questionary.Choice("Skip some file extensions", value="extensions"),
            questionary.Choice("Cancel", value="cancel"),
        ],
    ).ask()
    if choice in (None, "cancel"):
        return None
    if choice == "all":
        return {}
    if choice == "kinds":
        picked = questionary.checkbox(
            "File types to embed:",
            choices=[
                questionary.Choice(f"{kind}  {entry['files']} file(s), {human_bytes(entry['bytes'])}", value=kind, checked=True)
                for kind, entry in sorted(kinds.items(), key=lambda pair: KINDS.index(pair[0]))
            ],
        ).ask()
        if picked is None:
            return None
        return {"kinds": picked} if len(picked) < len(kinds) else {}
    ranked = sorted(extensions.values(), key=lambda entry: -entry["files"])
    picked = questionary.checkbox(
        "Extensions to skip:",
        choices=[
            questionary.Choice(f"{entry['ext']}  {entry['files']} file(s), {human_bytes(entry['bytes'])}", value=entry["ext"])
            for entry in ranked
            if entry["ext"].startswith(".")
        ],
    ).ask()
    if picked is None:
        return None
    return {"skip_ext": picked} if picked else {}


def location_of(hit: dict[str, Any]) -> str:
    if hit["start_line"] is not None:
        if hit["start_line"] == hit["end_line"]:
            return f":{hit['start_line']}"
        return f":{hit['start_line']}-{hit['end_line']}"
    if hit["start_sec"] is not None:
        return f" @ {clock(hit['start_sec'])}–{clock(hit['end_sec'])}"
    return ""


def render_hits(result: dict[str, Any], lines: int) -> None:
    hits = result["hits"]
    if not hits:
        console.print("No matches.")
        return
    for position, hit in enumerate(hits, 1):
        header = Text()
        header.append(f"{position:>2}  ", style="dim")
        weak = hit["score"] is not None and result.get("calibrated") and hit["score"] < WEAK_SCORE
        header.append(f"{hit['score']:.3f}" if hit["score"] is not None else "  –  ", style="dim" if weak else "bold")
        header.append("  ")
        shown = display_path(hit["path"])
        header.append(shown + location_of(hit), style=f"bold {KIND_STYLE[hit['kind']]} link file://{hit['path']}")
        tags = [hit["kind"]]
        if hit["part"] == "audio" and hit["kind"] == "video":
            tags.append("soundtrack")
        if hit["keyword"] >= 0.5:
            tags.append("keyword match")
        if hit["stale"]:
            tags.append("changed since last sync")
        if not hit["exists"]:
            tags.append("file is gone")
        header.append("  " + " · ".join(tags), style="dim")
        console.print(header)
        if hit["text"] and lines:
            body = hit["text"].splitlines()
            start = hit["start_line"] or 1
            for offset, line in enumerate(body[:lines]):
                snippet = Text()
                snippet.append(f"      {start + offset:>5} ", style="dim")
                snippet.append(line[:160], style="dim")
                console.print(snippet, soft_wrap=True, no_wrap=True, overflow="ellipsis")
            if len(body) > lines:
                console.print(f"            [dim]… {len(body) - lines} more line(s)[/]")
    console.print(f"[dim]{len(hits)} result(s) from {result['searched']} parts in {result['seconds']:.2f}s[/]")


@app.command()
def search(
    query: Annotated[list[str] | None, typer.Argument(help="What you are looking for, in plain words.")] = None,
    limit: Annotated[int | None, typer.Option("--limit", "-n", help="Number of results")] = None,
    kind: Annotated[list[str] | None, typer.Option("--type", "-t", help="Only these types: code,text,image,video,audio")] = None,
    inside: Annotated[Path | None, typer.Option("--in", help="Only results under this directory or file")] = None,
    like: Annotated[Path | None, typer.Option("--like", help="Find things similar to this file instead of a text query")] = None,
    mode: Annotated[str | None, typer.Option(help="hybrid, vector or keyword")] = None,
    per_file: Annotated[int | None, typer.Option("--per-file", help="Maximum results from one file, 0 for no cap")] = None,
    min_score: Annotated[float | None, typer.Option("--min-score", help="Hide results below this similarity")] = None,
    lines: Annotated[int, typer.Option("--lines", "-l", help="Lines of text to preview per result")] = 6,
    paths_only: Annotated[bool, typer.Option("--paths", help="Print only the matching paths")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine readable output")] = False,
    open_first: Annotated[bool, typer.Option("--open", "-o", help="Open the best match with the default app")] = False,
) -> None:
    """Search everything you have embedded."""
    text = " ".join(query or []).strip()
    if not text and like is None:
        raise fail("give a search query, or --like <file>")
    if mode is not None and mode not in ("hybrid", "vector", "keyword"):
        raise fail("--mode must be hybrid, vector or keyword")
    kinds = split_values(kind) or None
    paths = get_paths()
    config = Config.load(paths.config)
    ensure_service(paths, config)
    result = remote(
        paths,
        "search",
        {
            "query": text or None, "like": resolve(like) if like else None, "kinds": kinds,
            "under": resolve(inside) if inside else None, "limit": limit, "mode": mode,
            "per_file": per_file, "min_score": min_score,
        },
    )
    if as_json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif paths_only:
        for hit in result["hits"]:
            sys.stdout.write(hit["path"] + location_of(hit) + "\n")
    else:
        render_hits(result, lines)
    if open_first and result["hits"]:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen([opener, result["hits"][0]["path"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            errors.print(f"[yellow]could not run {opener}[/]")


@app.command("list")
def list_tracked(as_json: Annotated[bool, typer.Option("--json", help="Print machine readable output")] = False) -> None:
    """List the directories and files being tracked."""
    paths = get_paths()
    info = offline_status(paths, Config.load(paths.config))
    if as_json:
        console.print_json(json.dumps(info["roots"]))
        return
    if not info["roots"]:
        console.print("Nothing is tracked yet. Run `semantix sync <directory>`.")
        return
    table = Table(box=None, padding=(0, 2), header_style="dim")
    for column in ("Location", "Files", "Contents", "Waiting", "Last sync", "Rules"):
        table.add_column(
            column,
            justify="right" if column in ("Files", "Waiting") else "left",
            overflow="fold",
            no_wrap=column in ("Files", "Waiting", "Last sync"),
        )
    for root in info["roots"]:
        contents = ", ".join(f"{count} {kind}" for kind, count in sorted(root["kinds"].items(), key=lambda pair: KINDS.index(pair[0])))
        selection = root["selection"] or {}
        rules = []
        if selection.get("kinds"):
            rules.append("only " + "/".join(selection["kinds"]))
        if selection.get("skip_ext"):
            rules.append("skip " + " ".join(selection["skip_ext"]))
        if selection.get("exclude"):
            rules.append("exclude " + " ".join(selection["exclude"]))
        table.add_row(
            display_path(root["path"]), str(root["files"]), contents or "–", str(root["waiting"] or "–"),
            ago(root["synced_at"]), "; ".join(rules) or "–",
        )
    console.print(table)


@app.command()
def forget(
    target: Annotated[Path, typer.Argument(help="A tracked location, or a directory or file inside one.")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
) -> None:
    """Remove a location from the index. Your files are never touched."""
    paths = get_paths()
    path = resolve(target)
    if not yes and interactive():
        import questionary

        if not questionary.confirm(f"Remove {display_path(path)} from the index?", default=False).ask():
            raise typer.Exit(1)
    result = with_engine(paths, "forget", {"path": path})
    if result["root_removed"]:
        console.print(f"Forgot {display_path(path)} ({result['files']} file(s)). It is no longer tracked.")
    else:
        console.print(
            f"Forgot {display_path(path)} ({result['files']} file(s)). "
            f"It stays excluded from {display_path(result['root'])} as {result['excluded']}."
        )


@app.command()
def check(target: Annotated[Path, typer.Argument(help="A file or directory to explain.")]) -> None:
    """Explain whether a path would be embedded, and why not if it would not."""
    paths = get_paths()
    info = with_engine(paths, "check", {"path": resolve(target)})
    shown = display_path(info["path"])
    if not info["exists"]:
        console.print(f"{shown} does not exist.")
        raise typer.Exit(1)
    reason = info["reason"]
    if reason == "ignored" and info.get("source"):
        source = info["source"]
        reason = f"ignored by {display_path(source) if source.startswith('/') else source}"
    verdict = "[green]included[/]" if info["included"] else f"[yellow]left out[/]: {reason}"
    console.print(f"{shown}  {verdict}")
    details = []
    if info["kind"]:
        details.append(f"type {info['kind']}")
    details.append(f"tracked under {display_path(info['root'])}" if info["root"] else "not inside a tracked location")
    if info["status"]:
        details.append(f"index state {info['status']}")
    if info["pinned"]:
        details.append("added explicitly, so ignore rules do not apply")
    console.print(f"[dim]{' · '.join(details)}[/]")
    if info["error"]:
        console.print(f"[red]last error:[/] {info['error']}")


def show_config(config: Config) -> None:
    table = Table(box=None, padding=(0, 2), header_style="dim")
    table.add_column("Setting", no_wrap=True)
    table.add_column("Value", no_wrap=True)
    table.add_column("Meaning", style="dim")
    for key, option in OPTIONS.items():
        value = config.get(key)
        shown = ",".join(value) if isinstance(value, list) else str(value).lower() if isinstance(value, bool) else str(value)
        changed = value != option.default
        table.add_row(key, f"[bold]{shown}[/]" if changed else shown, option.help)
    console.print(table)


@config_app.callback()
def config_main(context: typer.Context) -> None:
    if context.invoked_subcommand is None:
        show_config(Config.load(get_paths().config))


@config_app.command("get")
def config_get(key: str) -> None:
    """Print one setting."""
    try:
        value = Config.load(get_paths().config).get(key)
    except ConfigError as error:
        raise fail(str(error)) from None
    console.print(",".join(value) if isinstance(value, list) else str(value).lower() if isinstance(value, bool) else str(value))


def apply_config(paths: Paths, config: Config) -> None:
    config.save(paths.config)
    if client.ping(paths) is None:
        return
    result = remote(paths, "reload")
    if result.get("model_changed"):
        console.print("The running service is reloading the model with the new settings.")
    else:
        console.print("The running service picked up the change.")


def indexed_dimension(paths: Paths) -> int | None:
    if not paths.db.exists():
        return None
    from .store import Store

    store = Store(paths.db, readonly=True)
    try:
        return store.stored_dimension()
    finally:
        store.close()


@config_app.command("set")
def config_set(
    key: str,
    value: str,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False,
) -> None:
    """Change one setting, e.g. `semantix config set model.capabilities text,image`."""
    paths = get_paths().ensure()
    try:
        config = Config.load(paths.config)
        previous = config.get(key)
        updated = config.set(key, value)
    except ConfigError as error:
        raise fail(str(error)) from None
    if updated == previous:
        console.print(f"{key} is already {value}")
        return
    stored = indexed_dimension(paths)
    resets = stored is not None and (
        (key == "model.dimension" and updated > stored) or key == "model.id"
    )
    if resets:
        errors.print("[yellow]This change invalidates the stored vectors: every file must be embedded again.[/]")
        if not yes:
            if not interactive():
                raise fail("run again with --yes to confirm")
            import questionary

            if not questionary.confirm("Continue?", default=False).ask():
                raise typer.Exit(1)
    elif key == "model.dimension" and stored is not None:
        console.print(f"Stored vectors will be shortened from {stored} to {updated} dimensions, no embedding needed.")
    apply_config(paths, config)
    console.print(f"{key} = {','.join(updated) if isinstance(updated, list) else updated}")
    if resets:
        if client.ping(paths) is None:
            from .engine import Engine

            Engine(paths).close()
        console.print("Run `semantix sync` to embed everything again.")
    if key.startswith(("image.", "video.", "audio.", "index.chunk")):
        console.print("[dim]Applies to files embedded from now on. Use `semantix sync --force <path>` to redo existing ones.[/]")


@config_app.command("preset")
def config_preset(name: str) -> None:
    """Apply a speed/quality preset for images and video: quality, balanced or fast."""
    if name not in PRESETS:
        raise fail(f"unknown preset {name!r}, choose from {', '.join(PRESETS)}")
    paths = get_paths().ensure()
    config = Config.load(paths.config)
    for key, value in PRESETS[name].items():
        config.set(key, value)
        console.print(f"{key} = {value}")
    apply_config(paths, config)


@config_app.command("path")
def config_path() -> None:
    """Print where settings and ignore rules live."""
    paths = get_paths()
    console.print(f"settings      {paths.config}")
    console.print(f"global ignore {paths.ignore}")
    console.print(f"index         {paths.db}")
    console.print(f"model         {paths.models}")
    console.print(f"log           {paths.log}")
