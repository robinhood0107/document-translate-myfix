import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtWidgets

from controller import ComicTranslate
from app.controllers.projects import ProjectController
from app.ui.settings.settings_page import SettingsPage
from app.projects.project_state_v2 import (
    close_cached_connection,
    load_state_from_proj_file_v2,
    save_state_to_proj_file_v2,
)
from modules.utils.project_glossary import (
    format_project_glossary_prompt,
    normalize_project_glossary,
    project_glossary_sha256,
)


class _CloseProbe(ComicTranslate):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)


class _MessageBoxProbe:
    Question = 1

    class Icon:
        Question = 1

    class ButtonRole:
        AcceptRole = 1
        DestructiveRole = 2
        RejectRole = 3

    choice = "Cancel"
    last = None

    def __init__(self, _parent):
        self.text = ""
        self.buttons = {}
        _MessageBoxProbe.last = self

    def setIcon(self, _value):
        pass

    def setWindowTitle(self, _value):
        pass

    def setText(self, value):
        self.text = value

    def addButton(self, label, _role):
        button = object()
        self.buttons[label] = button
        return button

    def setDefaultButton(self, _button):
        pass

    def exec(self):
        pass

    def clickedButton(self):
        return self.buttons.get(self.choice)


class ProjectGlossaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _window_with_pending_glossary_draft(self, temp_dir: str | None = None):
        window = _CloseProbe()
        settings = SettingsPage(window)
        window.settings_page = settings
        window.project_kind = "single"
        window.project_file = None
        window.image_files = []
        window.project_glossary = []
        window.text_ctrl = SimpleNamespace(
            _commit_pending_text_command=mock.Mock()
        )
        window.mark_project_dirty = mock.Mock()
        window._batch_active = False
        window.open_workspace_ctrl = SimpleNamespace(active=False)
        window.has_unsaved_changes = mock.Mock(return_value=False)
        window._show_project_switch_locked_warning = mock.Mock()
        window.project_ctrl = SimpleNamespace(
            ensure_autosave_project_file_for_transition=mock.Mock(),
            thread_save_project=mock.Mock(return_value=True),
            shutdown_autosave=mock.Mock(),
            save_main_page_settings=mock.Mock(),
        )
        settings.get_export_settings = mock.Mock(
            return_value={"project_autosave_enabled": True}
        )
        settings.save_settings = mock.Mock()
        settings.set_project_glossary_settings(
            [{"source": "north", "translation": "북쪽"}]
        )
        page = settings.ui.project_page
        page.project_glossary_table.item(0, 1).setText("북부")
        window.has_unsaved_changes.return_value = False
        page.add_glossary_entry()
        page.project_glossary_table.item(1, 0).setText("west")
        if temp_dir is not None:
            window.temp_dir = temp_dir
            window.file_handler = SimpleNamespace(archive_info=[])
            window.image_ctrl = SimpleNamespace(cleanup=mock.Mock())
            window.shutdown = mock.Mock()
            window._skip_close_prompt = False
        return window

    def test_prompt_keeps_terms_as_json_and_empty_glossary_is_noop(self):
        glossary = [
            {"source": "Ash", "translation": "재"},
            {"source": "Snow", "translation": "눈"},
        ]

        self.assertEqual(format_project_glossary_prompt([]), "")
        encoded = json.dumps(
            glossary,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.assertIn(encoded, format_project_glossary_prompt(glossary))

    def test_identity_is_deterministic_and_tracks_prompt_order(self):
        glossary = [
            {"source": "Ash", "translation": "재"},
            {"source": "Snow", "translation": "눈"},
        ]
        encoded = json.dumps(glossary, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

        self.assertEqual(project_glossary_sha256(glossary), digest)
        self.assertNotEqual(
            project_glossary_sha256(glossary),
            project_glossary_sha256(list(reversed(glossary))),
        )

    def test_invalid_or_oversized_entries_are_rejected_without_truncation(self):
        with self.assertRaises(ValueError):
            normalize_project_glossary([{"source": "Ash", "translation": ""}])
        with self.assertRaises(ValueError):
            normalize_project_glossary([{"source": "  ", "translation": ""}])
        with self.assertRaises(ValueError):
            normalize_project_glossary([{"source": "x" * 257, "translation": "value"}])
        with self.assertRaises(ValueError):
            normalize_project_glossary([{"source": 1, "translation": "value"}])
        with self.assertRaises(ValueError):
            normalize_project_glossary(
                [
                    {"source": str(index), "translation": "value"}
                    for index in range(257)
                ]
            )

    def test_project_settings_table_edits_and_deletes_entries(self):
        settings = SettingsPage()
        settings.set_project_glossary_settings(
            [{"source": "north", "translation": "북쪽"}]
        )
        page = settings.ui.project_page
        self.assertEqual(
            page.project_glossary_table.accessibleName(),
            page.tr("Project Glossary"),
        )
        main = SimpleNamespace(
            project_glossary=[],
            image_files=["neutral-page.png"],
            project_file="neutral-project.ctpr",
            mark_project_dirty=mock.Mock(),
        )
        settings.main_window = main
        page.glossary_changed.emit()
        self.assertEqual(
            main.project_glossary,
            [{"source": "north", "translation": "북쪽"}],
        )
        page.add_glossary_entry()
        page.project_glossary_table.item(1, 0).setText("south")
        page.project_glossary_table.item(1, 1).setText("남쪽")
        self.assertEqual(
            settings.get_project_glossary_settings(),
            [
                {"source": "north", "translation": "북쪽"},
                {"source": "south", "translation": "남쪽"},
            ],
        )
        self.assertEqual(main.project_glossary, settings.get_project_glossary_settings())
        self.assertTrue(main.mark_project_dirty.called)
        self.assertNotIn("project_glossary", settings.get_all_settings())
        page.project_glossary_table.selectRow(1)
        page.delete_glossary_entries()
        self.assertEqual(
            settings.get_project_glossary_settings(),
            [{"source": "north", "translation": "북쪽"}],
        )

    def test_incomplete_typing_does_not_poison_autosave_or_hide_draft(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = str(Path(temp_dir) / "draft.ctpr")
            valid = [{"source": "north", "translation": "북쪽"}]
            settings = SettingsPage()
            main = SimpleNamespace(
                project_glossary=[],
                image_files=["neutral-page.png"],
                project_file=project_path,
                mark_project_dirty=mock.Mock(),
            )
            settings.main_window = main
            settings.set_project_glossary_settings(valid)
            settings.main_window.mark_project_dirty.reset_mock()
            page = settings.ui.project_page
            page.add_glossary_entry()
            page.project_glossary_table.item(1, 0).setText("west")

            self.assertEqual(main.project_glossary, valid)
            self.assertFalse(page.glossary_status_label.isHidden())
            self.assertIn(
                "whole glossary table is pending",
                page.glossary_status_label.text().lower(),
            )
            self.assertEqual(
                page.glossary_entries(),
                valid + [{"source": "west", "translation": ""}],
            )

            def project_context(project_glossary):
                return SimpleNamespace(
                    curr_img_idx=-1,
                    webtoon_mode=False,
                    image_files=[],
                    export_source_by_path={},
                    image_states={},
                    image_data={},
                    image_history={},
                    in_memory_history={},
                    current_history_index={},
                    displayed_images=set(),
                    loaded_images=[],
                    image_patches={},
                    project_glossary=project_glossary,
                    project_output_preferences={},
                    project_file=project_path,
                    project_kind="single",
                    settings_page=SimpleNamespace(
                        get_llm_settings=lambda: {"extra_context": ""},
                        get_project_checkpoint_settings=lambda: {"enabled": False},
                    ),
                    image_viewer=SimpleNamespace(webtoon_view_state={}),
                    batch_report_ctrl=SimpleNamespace(
                        export_latest_report_for_project=lambda: None,
                        import_latest_report_from_project=lambda *_args, **_kwargs: None,
                    ),
                    temp_dir=temp_dir,
                )

            try:
                controller = ProjectController.__new__(ProjectController)
                controller.main = project_context(main.project_glossary)
                save_state_to_proj_file_v2(controller.main, project_path)
                close_cached_connection(project_path)
                controller.save_project(project_path)
                restored = project_context([])
                load_state_from_proj_file_v2(restored, project_path)
                self.assertEqual(restored.project_glossary, valid)
                close_cached_connection(project_path)

                page.project_glossary_table.item(1, 1).setText("서쪽")
                self.assertTrue(page.glossary_status_label.isHidden())
                self.assertEqual(
                    main.project_glossary,
                    valid + [{"source": "west", "translation": "서쪽"}],
                )
                controller.main.project_glossary = main.project_glossary
                controller.save_project(project_path)
                restored_after_correction = project_context([])
                load_state_from_proj_file_v2(
                    restored_after_correction,
                    project_path,
                )
                self.assertEqual(
                    restored_after_correction.project_glossary,
                    main.project_glossary,
                )
            finally:
                close_cached_connection(project_path)

    def test_clean_transition_prompts_and_cancel_keeps_glossary_draft(self):
        from controller import ComicTranslate

        window = self._window_with_pending_glossary_draft()
        proceed = mock.Mock()
        _MessageBoxProbe.choice = "Cancel"
        with mock.patch("controller.QtWidgets.QMessageBox", _MessageBoxProbe):
            result = ComicTranslate._run_guarded_project_transition(window, proceed)

        self.assertFalse(result)
        proceed.assert_not_called()
        window.project_ctrl.ensure_autosave_project_file_for_transition.assert_not_called()
        self.assertEqual(window.mark_project_dirty.call_count, 1)
        self.assertTrue(window.settings_page.has_unpersisted_project_glossary_draft())
        self.assertEqual(
            window.project_glossary,
            [{"source": "north", "translation": "북부"}],
        )
        self.assertEqual(
            window.settings_page.ui.project_page.glossary_entries(),
            [
                {"source": "north", "translation": "북부"},
                {"source": "west", "translation": ""},
            ],
        )
        self.assertIn(
            "whole glossary table is pending",
            _MessageBoxProbe.last.text.lower(),
        )
        self.assertEqual(set(_MessageBoxProbe.last.buttons), {"Don't Save", "Cancel"})

    def test_clean_transition_requires_explicit_discard_choice(self):
        from controller import ComicTranslate

        window = self._window_with_pending_glossary_draft()
        proceed = mock.Mock()
        _MessageBoxProbe.choice = "Don't Save"
        with mock.patch("controller.QtWidgets.QMessageBox", _MessageBoxProbe):
            result = ComicTranslate._run_guarded_project_transition(window, proceed)

        self.assertTrue(result)
        proceed.assert_called_once_with()
        window.project_ctrl.ensure_autosave_project_file_for_transition.assert_not_called()
        self.assertEqual(set(_MessageBoxProbe.last.buttons), {"Don't Save", "Cancel"})

    def test_discarding_draft_keeps_transition_autosave_for_other_project_edits(self):
        from controller import ComicTranslate

        window = self._window_with_pending_glossary_draft()
        window.has_unsaved_changes.return_value = True
        proceed = mock.Mock()
        window.project_ctrl.thread_save_project.side_effect = (
            lambda *, post_save_callback: (post_save_callback(), True)[1]
        )
        _MessageBoxProbe.choice = "Don't Save"
        with mock.patch("controller.QtWidgets.QMessageBox", _MessageBoxProbe):
            result = ComicTranslate._run_guarded_project_transition(window, proceed)

        self.assertTrue(result)
        window.project_ctrl.ensure_autosave_project_file_for_transition.assert_called_once()
        window.project_ctrl.thread_save_project.assert_called_once()
        proceed.assert_called_once_with()
        self.assertEqual(
            window.project_glossary,
            [{"source": "north", "translation": "북부"}],
        )

    def test_failed_transition_after_discard_confirmation_keeps_draft(self):
        from controller import ComicTranslate

        window = self._window_with_pending_glossary_draft()
        proceed = mock.Mock(side_effect=RuntimeError("synthetic transition failure"))
        _MessageBoxProbe.choice = "Don't Save"
        with mock.patch("controller.QtWidgets.QMessageBox", _MessageBoxProbe):
            with self.assertRaisesRegex(RuntimeError, "synthetic transition failure"):
                ComicTranslate._run_guarded_project_transition(window, proceed)

        self.assertTrue(window.settings_page.has_unpersisted_project_glossary_draft())
        self.assertEqual(
            window.settings_page.ui.project_page.glossary_entries(),
            [
                {"source": "north", "translation": "북부"},
                {"source": "west", "translation": ""},
            ],
        )

    def test_app_close_cancel_preserves_glossary_draft(self):
        from controller import ComicTranslate

        with tempfile.TemporaryDirectory() as temp_dir:
            app_temp = Path(temp_dir) / "app-temp"
            app_temp.mkdir()
            window = self._window_with_pending_glossary_draft(str(app_temp))
            event = QtGui.QCloseEvent()
            _MessageBoxProbe.choice = "Cancel"
            with mock.patch("controller.QtWidgets.QMessageBox", _MessageBoxProbe):
                ComicTranslate.closeEvent(window, event)

            self.assertFalse(event.isAccepted())
            window.project_ctrl.shutdown_autosave.assert_not_called()
            self.assertTrue(window.settings_page.has_unpersisted_project_glossary_draft())
            self.assertEqual(set(_MessageBoxProbe.last.buttons), {"Don't Save", "Cancel"})

    def test_app_close_requires_explicit_discard_choice(self):
        from controller import ComicTranslate

        with tempfile.TemporaryDirectory() as temp_dir:
            app_temp = Path(temp_dir) / "app-temp"
            app_temp.mkdir()
            window = self._window_with_pending_glossary_draft(str(app_temp))
            event = QtGui.QCloseEvent()
            _MessageBoxProbe.choice = "Don't Save"
            with mock.patch("controller.QtWidgets.QMessageBox", _MessageBoxProbe):
                ComicTranslate.closeEvent(window, event)

            self.assertTrue(event.isAccepted())
            window.project_ctrl.shutdown_autosave.assert_called_once_with(clear_recovery=True)
            self.assertEqual(set(_MessageBoxProbe.last.buttons), {"Don't Save", "Cancel"})

    def test_background_autosave_saves_valid_glossary_once_while_draft_is_pending(self):
        from app.controllers.projects import ProjectController

        window = self._window_with_pending_glossary_draft()
        window.project_file = "neutral-project.ctpr"
        window.image_files = ["neutral-page.png"]
        window.project_kind = "single"
        window._batch_active = False
        window._dirty_revision = 7
        window.has_unsaved_changes.return_value = True
        window.run_threaded = mock.Mock()
        window.set_project_clean = mock.Mock()
        window.set_project_clean.side_effect = (
            lambda: setattr(window.has_unsaved_changes, "return_value", False)
        )

        controller = ProjectController.__new__(ProjectController)
        controller.main = window
        controller._autosave_save_pending = False
        controller._autosave_retrigger_requested = False
        controller.save_current_state = mock.Mock()
        controller._ensure_autosave_project_file_if_needed = mock.Mock()
        controller.clear_recovery_checkpoint = mock.Mock()
        controller.add_recent_project = mock.Mock()
        controller._refresh_home_screen = mock.Mock()
        controller.save_project = mock.Mock()

        controller.autosave_project()

        window.run_threaded.assert_called_once()
        self.assertEqual(
            window.project_glossary,
            [{"source": "north", "translation": "북부"}],
        )
        self.assertTrue(window.settings_page.has_unpersisted_project_glossary_draft())
        self.assertEqual(window.mark_project_dirty.call_count, 1)
        window.run_threaded.call_args.args[3]()

        window.settings_page.ui.project_page.project_glossary_table.item(1, 0).setText(
            "westward"
        )
        self.assertEqual(window.mark_project_dirty.call_count, 1)
        window.has_unsaved_changes.return_value = False
        controller.autosave_project()

        self.assertEqual(window.run_threaded.call_count, 1)
        self.assertEqual(
            window.project_glossary,
            [{"source": "north", "translation": "북부"}],
        )

    def test_compiled_languages_translate_project_glossary_label(self):
        translated_discard_buttons = {
            "de": "Nicht speichern",
            "es": "No guardar",
            "fr": "Ne pas enregistrer",
            "it": "Non salvare",
            "ja": "保存しないでください",
            "ko": "저장 안 함",
            "ru": "Не сохранять",
            "tr": "Kaydetme",
            "zh-CN": "不保存",
        }
        translated_messages = {
            "de": (
                "Projektglossar",
                "Öffnen Sie ein Serienelement, um dessen Projektglossar zu bearbeiten.",
                "Die gesamte Glossartabelle bleibt ungespeichert, bis alle Zeilen vollständig sind. Füllen Sie jede Zeile aus, um sie zu speichern; andernfalls bleibt die letzte gültige Tabelle erhalten und diese Änderungen werden verworfen.",
            ),
            "es": (
                "Glosario del proyecto",
                "Abre un elemento de la serie para editar su glosario.",
                "Toda la tabla del glosario queda pendiente y no se guarda hasta que todas las filas estén completas. Completa todas las filas para guardarla; de lo contrario, se conservará la última tabla válida y se descartarán estos cambios.",
            ),
            "fr": (
                "Glossaire du projet",
                "Ouvrez un élément de la série pour modifier son glossaire de projet.",
                "L'ensemble du glossaire reste en attente et n'est pas enregistré tant que chaque ligne n'est pas complète. Complétez chaque ligne pour l'enregistrer ; sinon, la dernière version valide sera conservée et ces modifications seront abandonnées.",
            ),
            "it": (
                "Glossario del progetto",
                "Apri un elemento della serie per modificare il glossario del progetto.",
                "L'intera tabella del glossario resta in sospeso e non viene salvata finché tutte le righe non sono complete. Completa ogni riga per salvarla; altrimenti resterà l'ultima tabella valida e queste modifiche verranno scartate.",
            ),
            "ja": (
                "プロジェクト用語集",
                "プロジェクト用語集を編集するには、シリーズ項目を開いてください。",
                "すべての行が完成するまで、用語集テーブル全体は保留され保存されません。保存するにはすべての行を完成させてください。そうしない場合、最後に有効だった表が保持され、今回の編集は破棄されます。",
            ),
            "ko": (
                "프로젝트 용어집",
                "프로젝트 용어집을 편집하려면 시리즈 항목을 여세요.",
                "모든 행이 완성될 때까지 용어집 표 전체가 보류되어 저장되지 않습니다. 저장하려면 모든 행을 완성하세요. 그렇지 않으면 마지막으로 유효했던 표가 유지되고 이번 편집은 버려집니다.",
            ),
            "ru": (
                "Глоссарий проекта",
                "Откройте элемент серии, чтобы изменить глоссарий проекта.",
                "Вся таблица глоссария ожидает завершения и не сохраняется, пока не заполнены все строки. Заполните все строки, чтобы сохранить таблицу; иначе останется последняя действительная версия, а эти изменения будут отброшены.",
            ),
            "tr": (
                "Proje Sözlüğü",
                "Proje sözlüğünü düzenlemek için bir seri öğesi açın.",
                "Tüm sözlük tablosu beklemededir ve bütün satırlar tamamlanana kadar kaydedilmez. Kaydetmek için tüm satırları tamamlayın; aksi hâlde son geçerli tablo korunur ve bu değişiklikler atılır.",
            ),
            "zh-CN": (
                "项目词汇表",
                "打开系列项目以编辑其项目词汇表。",
                "所有行完整之前，整个词汇表都处于待保存状态。请填写所有行以保存；否则将保留上次有效的词汇表并丢弃这些编辑。",
            ),
        }
        app = QtWidgets.QApplication.instance()
        assert app is not None
        translations_dir = (
            Path(__file__).resolve().parents[1]
            / "resources"
            / "translations"
            / "compiled"
        )

        for language, expected in translated_messages.items():
            translator = QtCore.QTranslator()
            qm_path = translations_dir / f"ct_{language}.qm"
            self.assertTrue(translator.load(str(qm_path)))
            app.installTranslator(translator)
            try:
                for source, translation in zip(
                    (
                        "Project Glossary",
                        "Open a series item to edit its project glossary.",
                        "The whole glossary table is pending and is not saved until every row is complete. Complete every row to save it; otherwise, saving keeps the last valid table and discards these edits.",
                    ),
                    expected,
                ):
                    self.assertEqual(
                        QtCore.QCoreApplication.translate(
                            "ProjectPage",
                            source,
                        ),
                        translation,
                    )
                self.assertEqual(
                    QtCore.QCoreApplication.translate(
                        "ComicTranslate",
                        "Don't Save",
                    ),
                    translated_discard_buttons[language],
                )
            finally:
                app.removeTranslator(translator)


if __name__ == "__main__":
    unittest.main()
