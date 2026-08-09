from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QT_TRANSLATE_NOOP


@dataclass(frozen=True)
class ShortcutDefinition:
    id: str
    label: str
    description: str
    default: str


SHORTCUT_DEFINITIONS: tuple[ShortcutDefinition, ...] = (
    ShortcutDefinition(
        id="undo",
        label=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Undo"),
        description=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Undo the last editing action."),
        default="Ctrl+Z",
    ),
    ShortcutDefinition(
        id="redo",
        label=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Redo"),
        description=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Redo the previously undone action."),
        default="Ctrl+Y",
    ),
    ShortcutDefinition(
        id="save_project",
        label=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Save Project"),
        description=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Save editable state and update dirty render output."),
        default="Ctrl+S",
    ),
    ShortcutDefinition(
        id="delete_selected_box",
        label=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Delete Selected Box"),
        description=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Delete the currently selected text or block box."),
        default="Delete",
    ),
    ShortcutDefinition(
        id="series_back",
        label=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Back to Series Board"),
        description=QT_TRANSLATE_NOOP(
            "ShortcutDefinitions",
            "Leave the current series chapter and go back to the series board.",
        ),
        default="Alt+Left",
    ),
    ShortcutDefinition(
        id="restore_text_blocks",
        label=QT_TRANSLATE_NOOP("ShortcutDefinitions", "Restore Text Blocks"),
        description=QT_TRANSLATE_NOOP(
            "ShortcutDefinitions",
            "Draw saved text blocks back onto the image for editing.",
        ),
        default="Ctrl+Shift+R",
    ),
)


def get_shortcut_definitions() -> tuple[ShortcutDefinition, ...]:
    return SHORTCUT_DEFINITIONS


def get_default_shortcuts() -> dict[str, str]:
    return {definition.id: definition.default for definition in SHORTCUT_DEFINITIONS}
