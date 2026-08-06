"""설정은 사라져서는 안 된다.

사용자가 같은 설정을 몇 번이나 다시 하게 된 원인을 전수 조사해 특정했다. 저장 위치는
정상이었다 — 모든 호출부가 같은 `QSettings("ComicLabs", "ComicTranslate")` 를 쓰고
`settings.clear()` 는 어디에도 없다. 원인은 전부 쓰기 경로 로직이었고, 여기서 그것들을
계약으로 고정한다.

* 설정 페이지에 **없는 위젯**(`color_button`)을 읽어, 저장할 때마다 본문 색을
  `#000000` 으로 확정했다.
* 앱이 **시작할 때 스스로 전체 설정을 저장**했다. 아직 채워지지 않은 위젯의 생성자
  기본값이 그때 디스크에 박힌다. `min_font_size` 25 → 9 가 그 경로다.
* 서식 도구모음이 쓰는 12개 키는 **정상 종료에서만** 저장됐다.
* 콤보에 없는 이름이 저장돼 있으면 선택이 **빈 문자열로 지워졌다.**
* `credentials` 그룹이 저장마다 **통째로 삭제**됐다.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings import settings_page as settings_page_module  # noqa: E402
from app.ui.settings.settings_page import (  # noqa: E402
    SettingsPage,
    TEXT_RENDERING_CLOBBERED_COLOR,
    TEXT_RENDERING_CLOBBERED_OUTLINE_COLOR,
    TEXT_RENDERING_COLOR_KEY,
    TEXT_RENDERING_COLOR_RECOVERY_VERSION,
    TEXT_RENDERING_COLOR_RECOVERY_VERSION_KEY,
    TEXT_RENDERING_OUTLINE_COLOR_KEY,
    migrate_clobbered_text_rendering_colors,
    should_write_setting_value,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeQSettings:
    """마이그레이션 테스트용 최소 대역. 기존 마이그레이션 테스트와 같은 방식이다."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})
        self.synced = 0

    def value(self, key, default=None, type=None):  # noqa: A002 - QSettings 시그니처
        raw = self.values.get(key, default)
        if type is int:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 0
        if type is str:
            return "" if raw is None else str(raw)
        if type is bool:
            return bool(raw)
        return raw

    def setValue(self, key, value) -> None:
        self.values[key] = value

    def remove(self, key) -> None:
        self.values.pop(key, None)

    def sync(self) -> None:
        self.synced += 1


class NeverWriteAnUncertainValueTests(unittest.TestCase):
    """확신 없는 값으로 저장된 선택을 덮지 않는다."""

    def test_none_is_never_written(self) -> None:
        # `None` 은 잘못된 QVariant 가 된다. 어떤 키에도 쓰지 않는다.
        for key in ("extra_context", "font_family", "api_key"):
            self.assertFalse(should_write_setting_value(key, None))

    def test_a_blank_selection_never_overwrites(self) -> None:
        for key in ("font_family", "translator", "ocr", "color", "outline_color"):
            with self.subTest(key=key):
                self.assertFalse(should_write_setting_value(key, ""))
                self.assertFalse(should_write_setting_value(key, "   "))

    def test_a_blank_text_field_is_still_writable(self) -> None:
        """빈 값이 정당한 키를 막으면 사용자가 지울 수 없게 된다."""

        for key in ("extra_context", "completion_sound_file", "ntfy_topic"):
            with self.subTest(key=key):
                self.assertTrue(should_write_setting_value(key, ""))

    def test_real_values_pass(self) -> None:
        self.assertTrue(should_write_setting_value("font_family", "Meiryo"))
        self.assertTrue(should_write_setting_value("min_font_size", 0))
        self.assertTrue(should_write_setting_value("upper_case", False))

    def test_both_save_paths_apply_the_rule(self) -> None:
        """`text_rendering` 을 쓰는 경로가 둘이므로 양쪽 다 걸러야 한다."""

        from app.controllers.projects import ProjectController

        for func in (SettingsPage.save_settings, ProjectController.process_group):
            with self.subTest(func=func.__qualname__):
                self.assertIn(
                    "should_write_setting_value",
                    inspect.getsource(func),
                )


class ColorButtonsMustBeRealWidgetsTests(unittest.TestCase):
    """없는 위젯을 읽어 기본값을 확정하던 경로."""

    def test_the_settings_ui_is_no_longer_asked_for_color_buttons(self) -> None:
        source = inspect.getsource(SettingsPage.get_text_rendering_settings)
        self.assertNotIn('getattr(self.ui, "color_button"', source)
        self.assertNotIn('getattr(self.ui, "outline_color_button"', source)

    def test_the_real_owner_is_the_main_window(self) -> None:
        source = inspect.getsource(SettingsPage._color_button)
        self.assertIn("self.window()", source)

    def test_the_dead_color_picker_is_gone(self) -> None:
        """도달하면 AttributeError 가 날 코드였다."""

        self.assertFalse(hasattr(SettingsPage, "select_color"))

    def test_an_unreadable_color_writes_nothing(self) -> None:
        """색을 읽지 못하면 키를 빼야 한다. 기본값으로 덮는 것보다 언제나 옳다."""

        source = inspect.getsource(SettingsPage.get_text_rendering_settings)
        # 두 색은 리터럴 딕셔너리에 들어 있지 않고, 값을 읽은 뒤 조건부로 추가된다.
        for key in ("color", "outline_color"):
            with self.subTest(key=key):
                self.assertIn(f'settings["{key}"] =', source)
        self.assertIn("if color:", source)
        self.assertIn("if outline_color:", source)

    def test_no_setting_is_built_from_a_missing_settings_widget(self) -> None:
        """같은 사고의 재발 방지. 설정 값을 `getattr(self.ui, …, None)` 로 만들지 않는다."""

        source = pathlib.Path(settings_page_module.__file__).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("get_"):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                if not isinstance(inner.func, ast.Name):
                    continue
                if inner.func.id != "getattr" or len(inner.args) != 3:
                    continue
                target = inner.args[0]
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "ui"
                    and isinstance(inner.args[2], ast.Constant)
                    and inner.args[2].value is None
                ):
                    # 값을 만드는 데 쓰이면 조용히 기본값이 확정된다. 존재 여부
                    # 확인용(`is not None`)은 위 조건과 달라 여기 걸리지 않는다.
                    offenders.append(f"{node.name}: {ast.unparse(inner)}")
        allowed = {
            "get_all_settings: getattr(self.ui, 'project_checkpoint_enabled_checkbox', None)",
        }
        self.assertEqual(set(offenders) - allowed, set())


class StartupMustNotWriteSettingsTests(unittest.TestCase):
    """앱을 켰다 끄는 것만으로 저장된 설정이 바뀌어서는 안 된다."""

    def test_the_startup_sequence_is_wrapped(self) -> None:
        import controller

        source = pathlib.Path(controller.__file__).read_text(encoding="utf-8")
        self.assertIn("begin_external_load()", source)
        self.assertIn("end_external_load()", source)
        # 감싸는 순서가 핵심이다. `load_main_page_settings` 가 설정 페이지 위젯을
        # 건드리므로, 그것보다 먼저 가드가 서 있어야 한다.
        self.assertLess(
            source.index("begin_external_load()"),
            source.index("load_main_page_settings()"),
        )
        self.assertLess(
            source.index("load_settings()"),
            source.index("end_external_load()"),
        )

    def test_leaving_the_window_cancels_a_pending_save(self) -> None:
        source = inspect.getsource(SettingsPage.end_external_load)
        self.assertIn("_settings_save_timer.stop()", source)

    def test_both_debounced_savers_honour_the_guard(self) -> None:
        from app.controllers.projects import ProjectController

        self.assertIn(
            "_suppress_live_save",
            inspect.getsource(SettingsPage._save_settings_if_not_loading),
        )
        self.assertIn(
            "_suppress_live_save",
            inspect.getsource(SettingsPage._flush_scheduled_settings_save),
        )
        self.assertIn(
            "_live_save_suppressed",
            inspect.getsource(ProjectController.schedule_main_page_settings_save),
        )

    def test_the_main_page_timer_is_cancelled_too(self) -> None:
        import controller

        source = pathlib.Path(controller.__file__).read_text(encoding="utf-8")
        self.assertIn("cancel_scheduled_main_page_settings_save()", source)


class LeavingTheSettingsPageConfirmsTheWriteTests(unittest.TestCase):
    """250 ms 디바운스 안에서 페이지를 떠나면 마지막 변경이 사라졌다."""

    def test_the_flush_writes_and_clears_the_timer(self) -> None:
        source = inspect.getsource(SettingsPage.flush_pending_save)
        self.assertIn("_settings_save_timer.stop()", source)
        self.assertIn("self.save_settings()", source)
        # 로딩 중이면 저장하지 않는다 — 그 변경은 사용자 것이 아니다.
        self.assertIn("_suppress_live_save", source)

    def test_nothing_is_written_when_no_change_is_pending(self) -> None:
        source = inspect.getsource(SettingsPage.flush_pending_save)
        self.assertIn("if not self._settings_save_timer.isActive()", source)

    def test_navigating_away_triggers_the_flush(self) -> None:
        from app.ui.main_window import window as window_module

        source = pathlib.Path(window_module.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "currentChanged.connect(self._flush_settings_on_leave)",
            source,
        )
        handler = inspect.getsource(window_module.ComicTranslateUI._flush_settings_on_leave)
        # 설정 페이지에 머무는 전환에서는 아무것도 하지 않는다.
        self.assertIn("is settings_page", handler)
        self.assertIn("flush_pending_save()", handler)


class FormattingToolbarPersistsOnChangeTests(unittest.TestCase):
    """정상 종료에만 의존하면 크래시로 12개 키가 사라진다."""

    EXPECTED_WIDGETS = (
        "s_combo",
        "t_combo",
        "font_dropdown",
        "line_spacing_dropdown",
        "outline_width_dropdown",
        "block_font_color_button",
        "outline_font_color_button",
        "bold_button",
        "italic_button",
        "underline_button",
        "manual_radio",
        "automatic_radio",
        "outline_checkbox",
        "force_font_color_checkbox",
        "alignment_tool_group",
        "vertical_alignment_tool_group",
    )

    def test_every_formatting_widget_is_connected(self) -> None:
        from app.controllers.projects import ProjectController

        source = inspect.getsource(ProjectController.connect_main_page_persistence)
        for widget in self.EXPECTED_WIDGETS:
            with self.subTest(widget=widget):
                self.assertIn(f'"{widget}"', source)

    def test_the_connection_happens_after_the_ui_is_wired(self) -> None:
        """색 버튼은 `clicked` 를 공유한다. 색을 넣는 핸들러가 먼저 연결돼야 한다."""

        import controller

        source = pathlib.Path(controller.__file__).read_text(encoding="utf-8")
        self.assertLess(
            source.index("self.connect_ui_elements()"),
            source.index("connect_main_page_persistence()"),
        )

    def test_the_save_reaches_disk_immediately(self) -> None:
        from app.controllers.projects import ProjectController

        source = inspect.getsource(ProjectController.save_main_page_settings)
        self.assertIn("settings.sync()", source)

    def test_an_unmapped_language_does_not_abort_the_save(self) -> None:
        """`lang_mapping[...]` 의 KeyError 하나가 나머지 값까지 잃게 만들었다."""

        from app.controllers.projects import ProjectController

        source = inspect.getsource(ProjectController.save_main_page_settings)
        self.assertNotIn("lang_mapping[", source)
        self.assertIn("lang_mapping.get(", source)


class CredentialsGroupSurvivesTests(unittest.TestCase):
    def test_the_purge_only_runs_on_the_toggle_transition(self) -> None:
        source = inspect.getsource(SettingsPage.save_settings)
        # 그룹 전체 삭제는 사라졌다.
        self.assertNotIn('settings.remove("")', source)
        # 그리고 켜짐 → 꺼짐 전이일 때만 지운다.
        self.assertIn("was_saving_keys", source)

    def test_the_flag_itself_is_kept(self) -> None:
        source = inspect.getsource(SettingsPage.save_settings)
        self.assertIn('if key != "save_keys"', source)

    def test_a_missing_credential_widget_writes_nothing(self) -> None:
        source = inspect.getsource(SettingsPage.save_settings)
        self.assertIn("if value is None", source)


class GroupWipesAreAllowlistedTests(unittest.TestCase):
    """`remove("")` 는 그룹을 통째로 지운다. 쓸 곳이 정해져 있어야 한다."""

    ALLOWED = {"app/controllers/projects.py"}

    def test_only_the_recent_project_list_wipes_its_group(self) -> None:
        offenders: list[str] = []
        for path in REPO_ROOT.glob("app/**/*.py"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            if 'settings.remove("")' not in path.read_text(encoding="utf-8"):
                continue
            if relative not in self.ALLOWED:
                offenders.append(relative)
        self.assertEqual(offenders, [])


class TestsMustNotTouchTheUserStoreTests(unittest.TestCase):
    """테스트 스위트 자체가 사용자 설정을 지우고 있었다.

    여러 테스트가 `QSettings("ComicLabs", "ComicTranslate")` 를 그대로 열고, 일부는
    실제 윈도우를 만들어 로드/저장 경로를 태운다. Windows 에서 그 저장소는 사용자의
    레지스트리다. 실제로 이 스위트를 돌린 것만으로 `text_rendering` 의 본문 색 두 개가
    사라졌다. `tests/conftest.py` 가 프로세스 전체를 임시 디렉터리로 돌려 막는다.
    """

    def test_the_safety_net_is_session_wide_and_automatic(self) -> None:
        """개별 테스트가 기억해서 켜야 하는 안전장치는 반드시 잊힌다."""

        conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        self.assertIn('scope="session"', conftest)
        self.assertIn("autouse=True", conftest)

    def test_the_backup_is_restored_even_when_a_test_fails(self) -> None:
        conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        # 복원은 `finally` 안에서 일어나야 한다.
        self.assertIn("finally:", conftest)
        marker = conftest.index("finally:")
        self.assertIn('"import"', conftest[marker:])

    def test_new_keys_written_by_tests_are_cleaned_up(self) -> None:
        """되돌리기만 하면 테스트가 새로 만든 키가 남는다."""

        conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
        marker = conftest.index("finally:")
        self.assertIn('"delete"', conftest[marker:])

    def test_a_format_redirect_would_not_have_worked(self) -> None:
        """PySide6 의 두 인수 생성자는 defaultFormat 을 무시하고 레지스트리를 쓴다.

        이 계약은 "왜 임시 파일 방식이 아닌가"를 코드로 남겨, 나중에 누가 더
        간단해 보이는 방식으로 되돌리는 것을 막는다.
        """

        from PySide6.QtCore import QSettings

        if os.name != "nt":
            self.skipTest("레지스트리 저장소는 Windows 에서만 해당한다")
        previous = QSettings.defaultFormat()
        try:
            QSettings.setDefaultFormat(QSettings.Format.IniFormat)
            self.assertIn(
                "HKEY_CURRENT_USER",
                QSettings("ComicLabs", "ComicTranslate").fileName(),
            )
        finally:
            QSettings.setDefaultFormat(previous)


class HdStrategyIsWrittenWholeTests(unittest.TestCase):
    def test_all_three_values_are_always_saved(self) -> None:
        source = inspect.getsource(SettingsPage.get_hd_strategy_settings)
        for key in ("resize_limit", "crop_margin", "crop_trigger_size"):
            self.assertIn(key, source)
        # 전략별 분기가 사라졌다 — 반대편 값이 낡은 채로 남지 않는다.
        self.assertNotIn("elif", source)

    def test_developer_performance_mode_is_persisted_with_hd_strategy_settings(self) -> None:
        source = inspect.getsource(SettingsPage.get_hd_strategy_settings)
        self.assertIn("developer_performance_mode", source)

    def test_hd_strategy_performance_mode_has_read_path_symmetry(self) -> None:
        source = inspect.getsource(SettingsPage.load_settings)
        marker = source.index('beginGroup("hd_strategy")')
        end = source.index("settings.endGroup()", marker)
        block = source[marker:end]
        self.assertIn("developer_performance_mode", block)
        self.assertIn("_set_hd_strategy_performance_mode", block)

    def test_the_read_side_is_symmetric(self) -> None:
        source = inspect.getsource(SettingsPage.load_settings)
        marker = source.index('beginGroup("hd_strategy")')
        end = source.index("settings.endGroup()", marker)
        block = source[marker:end]
        self.assertIn("resize_spinbox.setValue", block)
        self.assertIn("crop_margin_spinbox.setValue", block)
        self.assertIn("crop_trigger_spinbox.setValue", block)


class ClobberedColorRecoveryTests(unittest.TestCase):
    def test_the_clobbered_values_are_cleared_once(self) -> None:
        settings = _FakeQSettings(
            {
                TEXT_RENDERING_COLOR_KEY: TEXT_RENDERING_CLOBBERED_COLOR,
                TEXT_RENDERING_OUTLINE_COLOR_KEY: (
                    TEXT_RENDERING_CLOBBERED_OUTLINE_COLOR
                ),
            }
        )
        self.assertTrue(migrate_clobbered_text_rendering_colors(settings))
        self.assertNotIn(TEXT_RENDERING_COLOR_KEY, settings.values)
        self.assertNotIn(TEXT_RENDERING_OUTLINE_COLOR_KEY, settings.values)
        self.assertEqual(
            settings.values[TEXT_RENDERING_COLOR_RECOVERY_VERSION_KEY],
            TEXT_RENDERING_COLOR_RECOVERY_VERSION,
        )
        self.assertEqual(settings.synced, 1)

    def test_a_real_choice_is_preserved(self) -> None:
        settings = _FakeQSettings(
            {
                TEXT_RENDERING_COLOR_KEY: "#ff0000",
                TEXT_RENDERING_OUTLINE_COLOR_KEY: "#00ff00",
            }
        )
        self.assertFalse(migrate_clobbered_text_rendering_colors(settings))
        self.assertEqual(settings.values[TEXT_RENDERING_COLOR_KEY], "#ff0000")
        self.assertEqual(settings.values[TEXT_RENDERING_OUTLINE_COLOR_KEY], "#00ff00")

    def test_the_marker_blocks_a_second_pass(self) -> None:
        settings = _FakeQSettings(
            {
                TEXT_RENDERING_COLOR_RECOVERY_VERSION_KEY: (
                    TEXT_RENDERING_COLOR_RECOVERY_VERSION
                ),
                TEXT_RENDERING_COLOR_KEY: TEXT_RENDERING_CLOBBERED_COLOR,
            }
        )
        self.assertFalse(migrate_clobbered_text_rendering_colors(settings))
        self.assertEqual(
            settings.values[TEXT_RENDERING_COLOR_KEY],
            TEXT_RENDERING_CLOBBERED_COLOR,
        )

    def test_empty_settings_still_stamp_the_marker(self) -> None:
        settings = _FakeQSettings()
        self.assertFalse(migrate_clobbered_text_rendering_colors(settings))
        self.assertEqual(
            settings.values[TEXT_RENDERING_COLOR_RECOVERY_VERSION_KEY],
            TEXT_RENDERING_COLOR_RECOVERY_VERSION,
        )

    def test_the_recovery_runs_before_the_only_reader(self) -> None:
        from app.controllers.projects import ProjectController

        source = inspect.getsource(ProjectController.load_main_page_settings)
        self.assertLess(
            source.index("migrate_clobbered_text_rendering_colors(settings)"),
            source.index("beginGroup('text_rendering')"),
        )


class EveryMigrationSyncsTests(unittest.TestCase):
    """마커만 쓰고 sync 를 빼면 크래시 시 마이그레이션이 다시 돈다."""

    def test_no_migration_forgets_to_sync(self) -> None:
        source = pathlib.Path(settings_page_module.__file__).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        missing: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("migrate_"):
                continue
            body = ast.unparse(node)
            if "settings.sync()" not in body:
                missing.append(node.name)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
