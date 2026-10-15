import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from core import runner as runner_module
from core.runner import (
    MacroRunner,
    _team_equipment_checkbox_checked,
    _team_loadout_row_selected,
)


@pytest.fixture(autouse=True)
def _blank_game_capture(monkeypatch):
    monkeypatch.setattr(
        runner_module.vision,
        "capture_game_bgr",
        lambda _hwnd: np.zeros((756, 1152, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(runner_module.vision, "find_image", lambda *_args, **_kwargs: None)


def _runner():
    runner = MacroRunner(MagicMock(), MagicMock(), MagicMock())
    runner._coords.update({
        "team_loadout_x": 270,
        "team_loadout_y": 241,
        "team_loadout_row_height": 38,
        "team_equipment_x": 600,
        "team_equipment_y": 584,
        "team_use_x": 884,
        "team_use_y": 584,
    })
    return runner


def _checkbox_frame(checked):
    frame = np.zeros((756, 1152, 3), dtype=np.uint8)
    if checked:
        frame[579:590, 594:607] = (0, 255, 0)
    else:
        frame[579:590, 594:607] = (32, 32, 32)
    return frame


def _team_row_frame(selected, x=270, y=241):
    frame = np.zeros((756, 1152, 3), dtype=np.uint8)
    color = (0, 150, 255) if selected else (60, 60, 60)
    frame[y - 14:y + 15, x - 55:x + 55] = color
    return frame


def _selected_team_frame():
    frame = _team_row_frame(True)
    frame[579:590, 594:607] = (0, 255, 0)
    return frame


def _close_click_count(clicked_matches):
    return sum(
        isinstance(match, dict) and match.get("name") == "nav_closeui"
        for match in clicked_matches
    )


def _fast_clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(runner_module.time, "time", lambda: now[0])
    monkeypatch.setattr(
        runner_module.time, "sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )


def test_team_loadout_key_normalizes_slot_and_equipment(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(
        "core.templates.load_template",
        lambda _name: {"blocks": {"team": "2", "equipment": "exclude"}},
    )

    assert runner._team_loadout_key({"macro": "Farm"}) == (2, "exclude")


def test_checkbox_state_reader_detects_checked_and_unchecked():
    assert _team_equipment_checkbox_checked(_checkbox_frame(True), 600, 584) is True
    assert _team_equipment_checkbox_checked(_checkbox_frame(False), 600, 584) is False
    assert _team_equipment_checkbox_checked(None, 600, 584) is None


def test_team_row_selected_state_is_detected_from_highlight_color():
    assert _team_loadout_row_selected(_team_row_frame(True), 270, 241) is True
    assert _team_loadout_row_selected(_team_row_frame(False), 270, 241) is False
    assert _team_loadout_row_selected(None, 270, 241) is None


def test_successful_team_loadout_is_recorded_without_reopening_units(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    team_match = {"cx": 100, "cy": 100, "score": 0.95}
    monkeypatch.setattr(
        "core.templates.load_template",
        lambda _name: {"blocks": {"team": "2", "equipment": "include"}},
    )
    monkeypatch.setattr(
        runner_module.vision,
        "wait_for_image",
        lambda *_args, **_kwargs: {"cx": 100, "cy": 100, "score": 0.95},
    )
    monkeypatch.setattr(runner, "_apply_team_loadout_panel", lambda *_args: True)
    monkeypatch.setattr(runner, "_interruptible_sleep", lambda *_args: None)
    monkeypatch.setattr(
        runner_module.vision, "find_image",
        lambda _hwnd, name: (
            team_match if name == "team" and runner._keyboard.tap.call_count == 1 else None
        ),
    )

    assert runner._apply_team_loadout(123, stop, {"macro": "Farm"}) is True
    assert runner._last_applied_team_loadout == (2, "include")
    assert runner._keyboard.tap.call_args_list == [
        ((ord("H"),), {}), ((ord("H"),), {}),
    ]


def test_retries_closing_units_panel_until_image_disappears(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    team_match = {"score": 0.95}
    _fast_clock(monkeypatch)

    monkeypatch.setattr(
        runner_module.vision, "find_image",
        lambda _hwnd, name: (
            team_match if name == "team" and runner._keyboard.tap.call_count < 2 else None
        ),
    )

    assert runner._close_unit_manager_panel(123, stop) is True
    assert runner._keyboard.tap.call_args_list == [
        ((ord("H"),), {}), ((ord("H"),), {}),
    ]
    assert any("retrying H" in entry.args[0] for entry in runner._log.call_args_list)


def test_failed_team_loadout_closes_units_and_clears_previous_success(monkeypatch):
    runner = _runner()
    runner._last_applied_team_loadout = (1, "include")
    stop = threading.Event()
    monkeypatch.setattr(
        "core.templates.load_template",
        lambda _name: {"blocks": {"team": "2", "equipment": "include"}},
    )
    monkeypatch.setattr(
        runner_module.vision,
        "wait_for_image",
        lambda *_args, **_kwargs: {"cx": 100, "cy": 100, "score": 0.95},
    )
    monkeypatch.setattr(runner, "_apply_team_loadout_panel", lambda *_args: False)
    monkeypatch.setattr(runner, "_interruptible_sleep", lambda *_args: None)

    assert runner._apply_team_loadout(123, stop, {"macro": "Farm"}) is False
    assert runner._last_applied_team_loadout is None
    assert runner._keyboard.tap.call_args_list == [((ord("H"),), {}), ((ord("H"),), {})]


def test_matching_team_configuration_is_reused_for_next_task(monkeypatch):
    runner = _runner()
    runner._last_applied_team_loadout = (2, "include")
    monkeypatch.setattr(runner, "_team_loadout_key", lambda _task: (2, "include"))
    monkeypatch.setattr(runner, "_apply_team_loadout", MagicMock())
    monkeypatch.setattr(runner, "_run_prestart_blocks", MagicMock())
    monkeypatch.setattr(runner, "_checkpoint", lambda _stop: False)
    monkeypatch.setattr(runner_module.camera, "run_camera_setup", lambda *_args, **_kwargs: None)

    assert runner._run_prestart(123, threading.Event(), {"macro": "Farm"}, {}) is True
    runner._apply_team_loadout.assert_not_called()
    assert any("Reusing Team Loadout 2" in entry.args[0] for entry in runner._log.call_args_list)


def test_retries_teams_until_unit_teams_screen_opens(monkeypatch):
    runner = _runner()
    clicked_teams = []
    saved_screenshots = []
    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        runner_module.vision, "capture_game_bgr",
        lambda _hwnd: pytest.fail("team detection must not invoke OCR capture"),
    )
    monkeypatch.setattr(
        runner_module.vision,
        "wait_for_image",
        lambda _hwnd, name, **_kwargs: {"score": 0.95} if name == "team" else None,
    )
    monkeypatch.setattr(
        runner_module.vision,
        "click_match",
        lambda *_args: clicked_teams.append("clicked"),
    )
    monkeypatch.setattr(
        runner, "_save_debug_screenshot_unconditional",
        lambda _hwnd, name: saved_screenshots.append(name),
    )

    assert runner._apply_team_loadout_panel(
        123, threading.Event(), {"cx": 100, "cy": 100, "score": 0.95}, 1, "include"
    ) is False
    assert clicked_teams == ["clicked"] * runner_module.TEAM_LOADOUT_OPEN_RETRY_ATTEMPTS
    assert saved_screenshots == ["team_loadout_open_failed"]
    runner._mouse.click.assert_not_called()


def test_retries_only_teams_button_until_unit_teams_is_detected(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    team_match = {"cx": 100, "cy": 100, "score": 0.95}
    use_match = {"score": 0.97}
    close_match = {"name": "nav_closeui"}
    opening_checks = iter([None, {"score": 0.96}])
    clicked_matches = []

    _fast_clock(monkeypatch)
    monkeypatch.setattr(runner_module.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    frames = iter([
        _team_row_frame(True),
        _checkbox_frame(True),
    ])
    monkeypatch.setattr(runner_module.vision, "capture_game_bgr", lambda _hwnd: next(frames))

    def wait_for_image(_hwnd, name, **_kwargs):
        if name == "team_loadout_open":
            return next(opening_checks)
        raise AssertionError(f"unexpected image wait: {name}")

    def find_image(_hwnd, name):
        if name == "team_use":
            return use_match
        if name == "nav_closeui":
            return close_match
        if name == "team_loadout_open":
            return {"score": 0.95} if _close_click_count(clicked_matches) == 0 else None
        return None

    monkeypatch.setattr(runner_module.vision, "wait_for_image", wait_for_image)
    monkeypatch.setattr(runner_module.vision, "find_image", find_image)
    monkeypatch.setattr(
        runner_module.vision, "click_match",
        lambda _mouse, _hwnd, match: clicked_matches.append(match),
    )

    assert runner._apply_team_loadout_panel(123, stop, team_match, 1, "include") is True
    assert clicked_matches.count(team_match) == 2
    assert runner._mouse.click.call_args_list[0].args == (270, 241)
    assert clicked_matches.count(use_match) == 1
    assert _close_click_count(clicked_matches) == 1


def test_retries_team_row_until_selected_highlight_is_detected(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    team_match = {"cx": 100, "cy": 100, "score": 0.95}
    opened_match = {"score": 0.96}
    use_match = {"score": 0.97}
    close_match = {"name": "nav_closeui"}
    clicked_matches = []
    frames = iter([
        *[_team_row_frame(False) for _ in range(6)],
        _team_row_frame(True),
        _checkbox_frame(True),
    ])

    _fast_clock(monkeypatch)
    monkeypatch.setattr(runner_module.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    monkeypatch.setattr(runner_module.vision, "capture_game_bgr", lambda _hwnd: next(frames))
    monkeypatch.setattr(
        runner_module.vision, "wait_for_image",
        lambda _hwnd, name, **_kwargs: opened_match if name == "team_loadout_open" else None,
    )
    def find_image(_hwnd, name):
        if name == "team_use":
            return use_match
        if name == "nav_closeui":
            return close_match
        if name == "team_loadout_open":
            return {"score": 0.95} if _close_click_count(clicked_matches) == 0 else None
        return None

    monkeypatch.setattr(runner_module.vision, "find_image", find_image)
    monkeypatch.setattr(
        runner_module.vision, "click_match",
        lambda _mouse, _hwnd, match: clicked_matches.append(match),
    )

    assert runner._apply_team_loadout_panel(123, stop, team_match, 1, "include") is True
    assert runner._mouse.click.call_args_list[:2] == [
        ((270, 241), {}), ((270, 241), {}),
    ]
    assert clicked_matches.count(use_match) == 1
    assert _close_click_count(clicked_matches) == 1


@pytest.mark.parametrize(
    ("team_num", "equipment", "initially_checked", "expected_clicks"),
    [
        (1, "include", True, [(270, 241), (884, 584)]),
        (8, "exclude", True, [(270, 507), (600, 584), (884, 584)]),
        (4, "include", False, [(270, 355), (600, 584), (884, 584)]),
    ],
)
def test_selects_team_toggles_saved_equipment_and_uses_team(
        monkeypatch, team_num, equipment, initially_checked, expected_clicks):
    runner = _runner()
    stop = threading.Event()
    team_match = {"cx": 100, "cy": 100, "score": 0.95}
    opened_match = {"cx": 275, "cy": 185, "score": 0.91}
    close_match = {"name": "nav_closeui"}
    clicked_matches = []
    frames = iter([
        _team_row_frame(True, y=241 + (team_num - 1) * 38),
        _checkbox_frame(initially_checked),
        _checkbox_frame(equipment == "include"),
    ])

    monkeypatch.setattr(runner_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runner_module.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    monkeypatch.setattr(runner_module.vision, "capture_game_bgr", lambda _hwnd: next(frames))
    monkeypatch.setattr(
        runner_module.vision, "click_match",
        lambda _mouse, _hwnd, match: clicked_matches.append(match),
    )
    monkeypatch.setattr(
        runner_module.vision,
        "wait_for_image",
        lambda _hwnd, name, **_kwargs: opened_match if name == "team_loadout_open" else None,
    )
    def find_image(_hwnd, name):
        if name == "team_loadout_open":
            return {"score": 0.95} if _close_click_count(clicked_matches) == 0 else None
        if name == "nav_closeui":
            return close_match
        return None

    monkeypatch.setattr(
        runner_module.vision, "find_image", find_image,
    )

    assert runner._apply_team_loadout_panel(
        123, stop, team_match, team_num, equipment) is True
    actual_clicks = [(c.args[0], c.args[1]) for c in runner._mouse.click.call_args_list]
    assert clicked_matches.count(team_match) == 1
    assert clicked_matches.count(close_match) == 1
    assert actual_clicks == expected_clicks


def test_team_loadout_failure_when_saved_equipment_state_cannot_be_read(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    screenshots = []
    _fast_clock(monkeypatch)
    frames = iter([_team_row_frame(True), np.zeros((756, 1152, 3), dtype=np.uint8)])
    monkeypatch.setattr(runner_module.vision, "capture_game_bgr", lambda _hwnd: next(frames))
    monkeypatch.setattr(
        runner_module.vision,
        "wait_for_image",
        lambda _hwnd, _name, **_kwargs: {"score": 0.95},
    )
    monkeypatch.setattr(runner_module.vision, "click_match", lambda *_args: None)
    monkeypatch.setattr(
        runner, "_save_debug_screenshot_unconditional",
        lambda _hwnd, name: screenshots.append(name),
    )

    assert runner._apply_team_loadout_panel(
        123, stop, {"cx": 100, "cy": 100, "score": 0.95}, 1, "include") is False
    assert screenshots == ["team_loadout_equipment_state_failed"]
    assert runner._mouse.click.call_args_list[0].args == (270, 241)


def test_retries_use_team_until_unit_teams_closes(monkeypatch):
    runner = _runner()
    stop = threading.Event()
    use_match = {"score": 0.97}
    close_match = {"name": "nav_closeui"}
    clicked_matches = []
    screenshots = []

    _fast_clock(monkeypatch)
    monkeypatch.setattr(runner_module.vision, "ref_to_screen", lambda _hwnd, x, y: (x, y))
    monkeypatch.setattr(
        runner_module.vision, "capture_game_bgr",
        lambda _hwnd: _selected_team_frame(),
    )
    monkeypatch.setattr(
        runner_module.vision, "wait_for_image",
        lambda _hwnd, name, **_kwargs: {"score": 0.95} if name == "team_loadout_open" else None,
    )

    def find_image(_hwnd, name):
        if name == "team_use":
            return use_match
        if name == "nav_closeui":
            return close_match
        if name == "team_loadout_open":
            return {"score": 0.95} if _close_click_count(clicked_matches) < 2 else None
        return None

    monkeypatch.setattr(runner_module.vision, "find_image", find_image)
    monkeypatch.setattr(
        runner_module.vision, "click_match",
        lambda _mouse, _hwnd, match: clicked_matches.append(match),
    )
    monkeypatch.setattr(
        runner, "_save_debug_screenshot_unconditional",
        lambda _hwnd, name: screenshots.append(name),
    )

    assert runner._apply_team_loadout_panel(
        123, stop, {"cx": 100, "cy": 100, "score": 0.95}, 1, "include"
    ) is True
    assert clicked_matches.count(use_match) == 1
    assert _close_click_count(clicked_matches) == 2
    assert screenshots == []
    assert any("retrying its close button" in entry.args[0]
               for entry in runner._log.call_args_list)


def test_debug_team_loadout_runs_without_prestart_or_battle(monkeypatch):
    runner = _runner()
    tested = []
    monkeypatch.setattr(runner_module.wm, "is_window", lambda _hwnd: True)
    monkeypatch.setattr(runner_module.wm, "show_window", lambda _hwnd: None)
    monkeypatch.setattr(runner_module.wm, "activate_window", lambda _hwnd: True)
    monkeypatch.setattr(runner_module.wm, "prevent_sleep", lambda: None)
    monkeypatch.setattr(runner_module.wm, "allow_sleep", lambda: None)
    monkeypatch.setattr(runner_module.vision, "close_mss", lambda: None)
    monkeypatch.setattr(
        runner, "_apply_team_loadout",
        lambda _hwnd, _stop, _task, team_override=None: tested.append(team_override) or True,
    )
    monkeypatch.setattr(runner, "_run_prestart_blocks", MagicMock())
    monkeypatch.setattr(runner, "_load_battle_blocks", MagicMock())

    runner._run_debug_test(lambda: 123, "team_loadout", "", 6, threading.Event())

    assert tested == [6]
    runner._run_prestart_blocks.assert_not_called()
    runner._load_battle_blocks.assert_not_called()


def test_debug_team_loadout_can_start_without_macro_operation(monkeypatch):
    runner = _runner()
    started = []

    class _Thread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            started.append(self.kwargs)

    monkeypatch.setattr(runner_module.threading, "Thread", _Thread)

    result = runner.start_debug_test(
        lambda: None, "team_loadout", "", coords={}, team_num=6)

    assert result == {"ok": True}
    assert started[0]["args"][1:4] == ("team_loadout", "", 6)


@pytest.mark.parametrize("team_num", [None, 0, 9, "bad"])
def test_debug_team_loadout_rejects_invalid_team(team_num):
    result = _runner().start_debug_test(
        lambda: None, "team_loadout", "Farm", coords={}, team_num=team_num)
    assert result == {"ok": False, "reason": "bad_team"}
