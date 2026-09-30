"""MkDocs hooks for assets that live outside the documentation source tree."""

from pathlib import Path

from mkdocs.structure.files import File


_EDITOR_SOURCE = Path("tools") / "config_editor.html"
_EDITOR_DESTINATION = "config_editor.html"


def on_page_markdown(markdown, **kwargs):
    """Resolve repository editor links to the copy bundled with the site."""
    return markdown.replace(
        "](../../tools/config_editor.html)", f"]({_EDITOR_DESTINATION})"
    )


def on_files(files, config):
    """Bundle the standalone graphical config editor with the built site."""
    repo_root = Path(config.config_file_path).resolve().parent.parent
    source = repo_root / _EDITOR_SOURCE
    if source.is_file():
        files.append(
            File.generated(
                config,
                _EDITOR_DESTINATION,
                abs_src_path=str(source),
            )
        )
    return files
