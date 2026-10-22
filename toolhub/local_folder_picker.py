"""Native local folder picker for desktop-hosted Streamlit tools."""

from pathlib import Path


class LocalFolderPickerError(RuntimeError):
    """The desktop folder picker could not be opened."""


def pick_local_folder(initial_directory: str | Path | None = None) -> str | None:
    """Return a selected local folder path, or None when the dialog is cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise LocalFolderPickerError(
            "The native folder picker requires a Python installation with Tk support."
        ) from exc

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            parent=root,
            title="Choose dataset folder",
            initialdir=str(initial_directory) if initial_directory else None,
            mustexist=True,
        )
    except tk.TclError as exc:
        raise LocalFolderPickerError(
            "The native folder picker is unavailable. Run Streamlit on the desktop "
            "where the dataset folder is accessible."
        ) from exc
    finally:
        if root is not None:
            root.destroy()

    return str(Path(selected).resolve()) if selected else None


def pick_local_file(
    initial_directory: str | Path | None = None,
    filetypes: tuple[tuple[str, str], ...] = (("All files", "*.*"),),
) -> str | None:
    """Return a selected local file path, or None when the dialog is cancelled."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise LocalFolderPickerError(
            "The native file picker requires a Python installation with Tk support."
        ) from exc

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            parent=root,
            title="Choose local YOLO-World weights",
            initialdir=str(initial_directory) if initial_directory else None,
            filetypes=filetypes,
        )
    except tk.TclError as exc:
        raise LocalFolderPickerError(
            "The native file picker is unavailable. Run Streamlit on the desktop "
            "where the model file is accessible."
        ) from exc
    finally:
        if root is not None:
            root.destroy()

    return str(Path(selected).resolve()) if selected else None
