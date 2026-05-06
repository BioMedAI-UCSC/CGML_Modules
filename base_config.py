"""
BaseConfig — unified argument / config-file handling.

Priority order (highest wins):
    explicit CLI arg  >  config file value  >  argparse default

Usage
-----
Replace the usual::

    args = parser.parse_args()

with::

    from module.base_config import BaseConfig
    cfg = BaseConfig(parser)
    # Access values as attributes:  cfg.batch, cfg.lr, …
    # Or get a plain Namespace for drop-in compatibility:
    args = cfg.as_namespace()

Config file (YAML or JSON)
--------------------------
Pass ``--config path/to/run.yaml`` (or ``-c …`` where available).
Any key whose name matches an argparse *dest* overrides the default for
that argument.  Extra keys (e.g. model-architecture settings already
present in train.py's config.yaml) are preserved and accessible via
``cfg['key']`` or ``cfg.as_dict()``.

Saving a resolved config
------------------------
Pass ``--save-config output.yaml`` to dump the fully-resolved
configuration (all defaults, file overrides, and CLI overrides merged)
to a file.  Useful for reproducibility — re-run with
``--config output.yaml`` to reproduce the exact same run.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Internal sentinel — marks "value was not set by the user"
# ---------------------------------------------------------------------------
class _Unset:
    """Unique sentinel that is never a legitimate argument value."""
    _instance: Optional["_Unset"] = None

    def __new__(cls) -> "_Unset":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"


_UNSET = _Unset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_dests(parser: argparse.ArgumentParser) -> list[str]:
    """Return all action dest names that are not SUPPRESS."""
    return [
        a.dest
        for a in parser._actions
        if a.dest != argparse.SUPPRESS
    ]


def _existing_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {opt for a in parser._actions for opt in a.option_strings}


def _detect_explicit_args(
    parser: argparse.ArgumentParser,
    argv: Optional[list[str]],
) -> set[str]:
    """
    Return the set of dest names whose values were explicitly supplied on
    the command line (as opposed to merely having a default).

    Technique: pre-populate a Namespace with the sentinel for every dest.
    argparse only overwrites a dest that is *already present* in the
    namespace when the corresponding flag/positional actually appears on the
    command line — it skips ``set_defaults`` for keys already in the
    namespace.  So any dest that is no longer ``_UNSET`` after parsing was
    explicitly provided.
    """
    sentinel_ns = argparse.Namespace(
        **{dest: _UNSET for dest in _collect_dests(parser)}
    )
    try:
        parsed, _ = parser.parse_known_args(argv, namespace=copy.copy(sentinel_ns))
    except SystemExit:
        return set()
    return {k for k, v in vars(parsed).items() if v is not _UNSET}


def _load_config_file(path: str) -> dict:
    """Load a YAML or JSON file and return its contents as a dict."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open() as fh:
        content = fh.read()
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise ImportError(
                "PyYAML is required to read .yaml config files: pip install pyyaml"
            )
        data = _yaml.safe_load(content)
    elif suffix == ".json":
        data = json.loads(content)
    else:
        # Unknown extension — try YAML first, fall back to JSON.
        if _HAS_YAML:
            try:
                data = _yaml.safe_load(content)
            except _yaml.YAMLError:
                data = json.loads(content)
        else:
            data = json.loads(content)
    return data if isinstance(data, dict) else {}


def _save_config_file(path: str, data: dict) -> None:
    """Write *data* to a YAML or JSON file, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        suffix = p.suffix.lower()
        if suffix == ".json":
            json.dump(data, fh, indent=2, default=str)
        else:
            if not _HAS_YAML:
                raise ImportError(
                    "PyYAML is required to write .yaml config files: pip install pyyaml"
                )
            _yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BaseConfig:
    """
    Merges a config file with command-line arguments.

    Parameters
    ----------
    parser:
        An already-populated ``argparse.ArgumentParser``.  The class adds
        ``--config`` / ``-c`` and ``--save-config`` to it automatically
        (skipping any that already exist).
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when ``None``.

    Notes
    -----
    *   Config-file keys that do not correspond to any argparse dest are
        still stored and accessible (useful for model-architecture settings
        that live alongside training hyper-parameters in the same YAML).
    *   ``--save-config`` is processed inside ``__init__`` so the file is
        written before the rest of the script runs.  The argument itself is
        excluded from the saved output to keep the dump clean.
    """

    _CONFIG_LONG  = "--config"
    _CONFIG_SHORT = "-c"
    _SAVE_LONG    = "--save-config"

    def __init__(
        self,
        parser: argparse.ArgumentParser,
        argv: Optional[list[str]] = None,
    ) -> None:
        self._parser = parser
        self._argv   = argv  # None → sys.argv[1:] (handled by argparse)

        # Inject --config / --save-config before any parsing happens so that
        # the sentinel-detection pass can see them too.
        self._inject_config_args()

        # --- Pass 1: detect which dests were explicitly set on the CLI ------
        explicit: set[str] = _detect_explicit_args(parser, argv)

        # --- Pass 2: normal parse to get defaults + CLI values --------------
        args: argparse.Namespace = parser.parse_args(argv)

        # --- Load config file -----------------------------------------------
        file_data: dict = {}
        config_path: Optional[str] = getattr(args, "config", None)
        if config_path:
            file_data = _load_config_file(config_path)

        # --- Merge: defaults → file → explicit CLI --------------------------
        # Start with every value argparse produced (defaults for non-explicit,
        # actual values for explicit).
        merged: dict = dict(vars(args))

        # Override with config-file values (middle priority).
        # Only keys that do NOT appear as explicit CLI args are overridden.
        for key, value in file_data.items():
            if key not in explicit:
                merged[key] = value

        # Explicit CLI args are already in `merged` via `vars(args)`.

        # Store internals
        self._merged:   dict     = merged
        self._explicit: set[str] = explicit
        self._file_data: dict    = file_data

        # --- Handle --save-config -------------------------------------------
        save_path: Optional[str] = merged.get("save_config")
        if save_path:
            self.save(save_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inject_config_args(self) -> None:
        """Add --config and --save-config to the parser if not present."""
        existing = _existing_option_strings(self._parser)

        if self._CONFIG_LONG not in existing:
            shorts = (
                [self._CONFIG_SHORT]
                if self._CONFIG_SHORT not in existing
                else []
            )
            self._parser.add_argument(
                *shorts,
                self._CONFIG_LONG,
                default=None,
                type=str,
                metavar="FILE",
                help=(
                    "Path to a YAML or JSON config file.  "
                    "Explicit CLI arguments override values from the file."
                ),
            )

        if self._SAVE_LONG not in existing:
            self._parser.add_argument(
                self._SAVE_LONG,
                default=None,
                type=str,
                metavar="FILE",
                help=(
                    "Save the fully-resolved configuration to FILE "
                    "(YAML or JSON determined by extension) then continue."
                ),
            )

    # ------------------------------------------------------------------
    # Attribute / dict-style access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Avoid infinite recursion for private attributes that haven't been
        # set yet (e.g. during __init__ before _merged is assigned).
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._merged[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__!r} has no configuration key {name!r}"
            )

    def __getitem__(self, key: str) -> Any:
        return self._merged[key]

    def __contains__(self, key: str) -> bool:
        return key in self._merged

    def __iter__(self) -> Iterator[str]:
        return iter(self._merged)

    def __len__(self) -> int:
        return len(self._merged)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._merged!r})"

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not present."""
        return self._merged.get(key, default)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        """Return a shallow copy of the merged configuration as a plain dict."""
        return dict(self._merged)

    def as_namespace(self) -> argparse.Namespace:
        """
        Return an ``argparse.Namespace`` populated with the merged values.

        This provides full drop-in compatibility with code that was written
        against a plain ``parser.parse_args()`` result::

            cfg  = BaseConfig(parser)
            args = cfg.as_namespace()   # existing code uses args.batch, etc.
        """
        return argparse.Namespace(**self._merged)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, *, exclude: Optional[set[str]] = None) -> None:
        """
        Write the resolved configuration to *path*.

        The file format is determined by the extension:
        ``.yaml`` / ``.yml`` → YAML, ``.json`` → JSON, anything else → YAML.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created if needed.
        exclude:
            Optional set of key names to omit from the saved output.
            ``save_config`` is always excluded to avoid self-referential loops.
        """
        _excluded = {"save_config"} | (exclude or set())
        data = {k: v for k, v in self._merged.items() if k not in _excluded}
        _save_config_file(path, data)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def was_explicit(self, key: str) -> bool:
        """Return True if *key* was explicitly supplied on the command line."""
        return key in self._explicit

    def from_file(self, key: str) -> bool:
        """Return True if *key* came from the config file (and was not overridden by CLI)."""
        return key in self._file_data and key not in self._explicit
