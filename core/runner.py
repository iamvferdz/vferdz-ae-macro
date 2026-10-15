"""The macro's actual run loop -- Dashboard > Start. Currently covers the
launch sequence (confirm the task queue isn't empty, confirm we're on the
lobby screen, open Play > Story), picking the first task's map and stage,
Pre Start (camera setup, Team Loadout including saved-equipment selection,
a per-map default walk, the task's Macro Operation template's Pre Start
blocks), pressing Start Game, and then
watching the battle for Battle-phase blocks (Upgrade/Sell/Auto Upgrade Unit)
and a Victory/Defeat screen -- which is recorded to run history/win-loss
counts and, if a webhook is configured, reported to Discord as a screenshot
of the result screen plus a rendered win/loss card (see
_handle_match_result/_send_result_webhook -- the stats/reward OCR this used
to do per match now lives only behind Settings > Debug's on-demand "Read
Rewards"/"Read Game Stats" buttons). The remaining Battle-phase block types
(Walk/Wait/Setting) plug in as they are implemented.
"""
import os
import threading
import time
import traceback
import webbrowser
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone

import cv2

from . import camera
from . import config
from . import keys
from . import ocr_windows
from . import paths
from . import stage_select
from . import vision
from . import wave as wave_module
from .diagnostics import FailureCategory, RecoveryAction, FailureReport, create_failure_report, save_failure_snapshot
from . import window as wm
from .runner_constants import *  # noqa: F401,F403 -- see runner_constants' docstring
from .runner_blocks import BlockOps
from .runner_bounty import BountyOps
from .runner_challenge import ChallengeOps
from .runner_crafting import CraftingOps
from .runner_expedition import ExpeditionOps
from .runner_fuel import FuelOps
from .runner_shop import ShopOps


MAX_EXTRACT_AFTER = 9999


def _open_deep_link(url: str) -> None:
    """Open an application deep link on Windows and non-Windows systems."""
    startfile = getattr(os, "startfile", None)
    if startfile is not None:
        startfile(url)
        return
    if not webbrowser.open(url):
        raise OSError(f"No application accepted the deep link: {url}")


def _parse_extract_after(value, default=1):
    """Return a safe Expedition extraction count.

    Task files and presets are user-editable JSON, so this value cannot be
    trusted to have passed through the number input in the UI. Python integers
    can hold an arbitrarily large integer, but the browser starts serializing
    absurd values in scientific notation. Preserve the user's intent to run
    for as long as possible by clamping valid oversized values to 9,999.
    """
    if isinstance(value, bool):
        return default
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return default
    if not parsed.is_finite() or parsed < 0 or parsed != parsed.to_integral_value():
        return default
    if parsed > MAX_EXTRACT_AFTER:
        return MAX_EXTRACT_AFTER
    return int(parsed)


def _team_equipment_checkbox_checked(frame, x: int, y: int):
    """Read the green checkmark from the saved-equipment checkbox."""
    if frame is None or frame.size == 0:
        return None
    x1, x2 = max(0, x - 10), min(frame.shape[1], x + 11)
    y1, y2 = max(0, y - 10), min(frame.shape[0], y + 11)
    checkbox = frame[y1:y2, x1:x2]
    if checkbox.size == 0:
        return None
    if int(checkbox.max()) < 20:
        return None
    hsv = cv2.cvtColor(checkbox, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 100, 100), (95, 255, 255))
    return cv2.countNonZero(green) >= 8


def _team_loadout_row_selected(frame, x: int, y: int):
    """Detect the gold/orange selected-state fill within one team row."""
    if frame is None or frame.size == 0:
        return None
    x1, x2 = max(0, x - 55), min(frame.shape[1], x + 55)
    y1, y2 = max(0, y - 14), min(frame.shape[0], y + 15)
    row = frame[y1:y2, x1:x2]
    if row.size == 0:
        return None
    hsv = cv2.cvtColor(row, cv2.COLOR_BGR2HSV)
    selected = cv2.inRange(hsv, (8, 100, 100), (38, 255, 255))
    return cv2.countNonZero(selected) >= 50


class MacroRunner(BountyOps, ChallengeOps, CraftingOps, FuelOps, ShopOps, ExpeditionOps, BlockOps):
    """One run's worth of state -- module-level singleton via main.Api, same
    pattern as core.paths._recorder, since only one run can realistically be
    active at a time (one physical game window, one macro)."""

    def __init__(self, mouse, keyboard, log, set_status=None, record_result=None,
                 get_challenge_settings=None, mark_challenge_stage_played=None, get_run_stats=None,
                 get_crafting_settings=None, set_crafting_count=None, get_bounty_settings=None,
                 set_bounty_remaining=None, get_fuel_settings=None,
                 mark_fuel_refill_result=None, get_hotkeys=None,
                 get_auto_shop_settings=None,
                 save_auto_shop_item_state=None, save_auto_shop_shop_state=None,
                 get_global_story_settings=None):
        self._mouse = mouse
        self._keyboard = keyboard
        self._log = log
        # Live click-point overrides -- replaced with the user's saved values
        # at the top of _run; defaults here so the Settings > Debug test
        # paths (which never go through _run) still resolve every key.
        self._coords = dict(DEFAULT_COORDS)
        # Set at the top of _run when nav_unitmanager is already visible
        # (Start pressed from inside a stage); consumed one-shot by
        # _run_task to skip the first task's lobby/stage entry.
        self._skip_first_task_setup = False
        # Team/equipment successfully applied by the most recent loadout
        # operation. A matching task can keep using it when the queue moves
        # to the next stage instead of reopening Team Loadout every time.
        self._last_applied_team_loadout = None
        # "Leave at Minute" battle block (see runner_blocks): battle clock +
        # the flag it sets when it leaves. Real values set per match in
        # _play_one_match; defaults here so the Settings > Debug battle test
        # (which never goes through _play_one_match) resolves them too.
        self._battle_started_at = 0.0
        self._battle_leave_requested = False
        # Which crop the Wait for Wave block reads its badge from. Expedition
        # renders it somewhere the shared box does not reach -- see
        # EXPEDITION_WAVE_REGION. Set per match in _play_one_match.
        self._wave_region = WAVE_REGION
        # Whether THIS match is an Expedition, and when a reward card was last
        # picked. Together they let Wait for Wave release on a gamemode that
        # has no wave counter -- see WAIT_WAVE_NO_COUNTER_SETTLE.
        self._is_expedition_match = False
        # Proof the battle is genuinely under way (cards drop for kills), and
        # the quiet-period clock the deferred placements wait on. Anything
        # that disrupts the board -- a card, a mid-run Start Game -- restarts
        # the clock. See WAIT_WAVE_NO_COUNTER_SETTLE.
        self._last_reward_card_at = 0.0
        self._last_board_disruption_at = 0.0
        self._battle_replayed = False
        # Expedition checkpoint engine choice (see the EXP_COLOR_* block) +
        # the sighting debounce clock it uses; the real values arrive via
        # start()/_run, these are just never-ran-yet defaults. Same for the
        # Expedition camera's O-zoom hold (Settings > Debug).
        self._expedition_color_buttons = True
        self._exp_last_sighting_at = 0.0
        self._exp_failed_extracts = 0
        # Consecutive polls that found a checkpoint Continue, and when that
        # run of them started. See EXPEDITION_STALL_TIMEOUT -- a checkpoint
        # that never clears would otherwise be re-clicked all the way to
        # MATCH_RESULT_TIMEOUT.
        self._exp_checkpoint_streak = 0
        self._exp_checkpoint_since = 0.0
        # Polls a popup/reward card took before the checkpoint could be read.
        # See EXPEDITION_INTERCEPT_TIMEOUT. _exp_clock_marked_at is what lets
        # the checkpoint clock be HELD across those polls rather than aged.
        self._exp_intercept_streak = 0
        self._exp_intercept_since = 0.0
        self._exp_clock_marked_at = 0.0
        self._expedition_camera_o_ms = 100.0
        # Wrapped to remember the most recent action text locally: the
        # stop path (_checkpoint) reports "Stopped. (was: <action>)" so a
        # user stopping a visibly-hung run gets told what it was stuck on
        # -- the runner never sees main.Api's own status dict, so it keeps
        # its own copy of just the action string.
        _raw_set_status = set_status or (lambda **kw: None)
        self._last_action = ""

        def _tracking_set_status(**kw):
            if "action" in kw:
                self._last_action = kw["action"]
            _raw_set_status(**kw)

        self._set_status = _tracking_set_status
        # Used only by opt-in progress notifications to describe what a
        # Challenge will resume after it finishes.
        self._active_task_progress = None
        self._current_task = None
        self._active_task = None
        # (result: "win"|"loss", map_name, duration_str) -> persists to run
        # history / win-loss counters (see main.Api._record_match_result).
        self._record_result = record_result or (lambda *a, **kw: None)
        # Challenge tab settings live in settings.json, owned by main.Api
        # (same reason record_result is a callback, not a direct import --
        # avoids core/ reaching back into main.py). None (the default, e.g.
        # in tests/CLI mode) just makes _run_challenges a no-op.
        self._get_challenge_settings = get_challenge_settings
        self._get_bounty_settings = get_bounty_settings
        self._get_global_story_settings = get_global_story_settings or (lambda: {})
        # Read when an Auto Upgrade Unit block runs so a key changed in
        # Settings is used without rebuilding the runner or embedding a
        # machine-specific key inside every exported macro template.
        self._get_hotkeys = get_hotkeys or (lambda: {})
        self._set_bounty_remaining = set_bounty_remaining or (lambda *a, **kw: None)
        self._mark_challenge_stage_played = mark_challenge_stage_played or (lambda *a, **kw: None)
        # Returns a fresh session/all-time win-loss + session_start + version
        # snapshot for the match-result webhook (see _send_result_webhook).
        # None (tests/CLI) just yields an empty snapshot, so those fields are
        # quietly omitted rather than the webhook failing.
        self._get_run_stats = get_run_stats or (lambda: {})
        # Auto Crafting tab settings + its persisted win counter, same
        # callback pattern as Challenge above (settings.json is owned by
        # main.Api). None (tests/CLI) makes every crafting path a no-op.
        self._get_crafting_settings = get_crafting_settings
        self._set_crafting_count = set_crafting_count or (lambda *a, **kw: None)
        # Auto Fuel uses the same callback boundary as Challenge and Crafting:
        # main.Api owns settings.json, while the runner only reads normalized
        # state and reports each station's result.
        self._get_fuel_settings = get_fuel_settings
        self._mark_fuel_refill_result = mark_fuel_refill_result
        self._get_auto_shop_settings = get_auto_shop_settings
        self._save_auto_shop_item_state = save_auto_shop_item_state
        self._save_auto_shop_shop_state = save_auto_shop_shop_state
        self._thread = None
        self._stop_event = None
        self._pause_event = threading.Event()
        self._paused_logged = False
        self._stop_logged = False  # one "Stopped." per stop -- see _checkpoint
        self._debug_screenshots = False
        self._current_hwnd = None       # set at the top of _run
        self._hwnd_getter = None        # set at the top of _run -- lets _attempt_rejoin find a re-docked hwnd
        # A deep-link launch is handed to Roblox/Bloxstrap, so there is no
        # process handle for the runner to wait on or cancel. Keep the launch
        # section serialized anyway: a manual Force Rejoin or another recovery
        # path must never start a second launcher while the first one is still
        # waiting on Roblox's single-instance handoff.
        self._rejoin_lock = threading.Lock()
        self._rejoin_pending = False
        self._left_stage_this_run = False
        # Placed-unit screen positions from THIS match's Pre Start (see
        # _run_place_unit_block), keyed by the unit's #ordinal among place_unit
        # blocks (same numbering ui/app.js's listPlacedUnits() uses for the
        # Upgrade/Sell Unit pickers) -- lets Battle-phase Upgrade/Sell Unit
        # blocks click the right spot without needing their own recorded
        # position. Only overwritten when a placement actually runs, so a
        # block skipped via "Once" on a repeat keeps whatever position its
        # first placement recorded rather than losing it.
        self._placed_unit_positions = {}
        # Whether Left Shift is currently being held down for a "quick
        # place" chain (see _run_place_unit_block) -- true from right
        # before the FIRST of a run of consecutive same-unit Place Unit
        # blocks is clicked, through the last one, then released. Reset
        # False any time a run/test starts fresh so a leftover held key
        # from an interrupted previous run can never bleed into a new one.
        self._quick_place_shift_down = False
        # The running #ordinal counter place_unit blocks share -- Pre Start
        # blocks number first, Battle-phase place_unit blocks (see
        # _run_battle_blocks_tick) continue counting from wherever Pre
        # Start left off, matching ui/app.js's listPlacedUnits() (which
        # numbers place_unit blocks across BOTH phases in one sequence).
        # Reset to 0 once per match in _run_prestart.
        self._last_unit_ordinal = 0
        # Battle-phase block state (see _run_battle_blocks_tick) -- reset at
        # the start of each match in _play_one_match.
        self._battle_block_index = 0
        self._battle_block_state = {}
        # How many times exp_extract has shown up THIS match, and which
        # sighting actually accepts it (see _check_expedition_wave_result)
        # -- both reset alongside the battle block state in _play_one_match.
        self._expedition_extract_count = 0
        self._expedition_extract_accept_at = 1
        # Consecutive-loss fail-safe (see MAX_CONSECUTIVE_LOSSES_SAME_MAP):
        # how many losses in a row on _consecutive_loss_map so far -- reset
        # to 0 on any win, or restarted at 1 for a new map, so only a real
        # unbroken streak on the SAME map ever counts toward it.
        self._consecutive_losses = 0
        self._consecutive_loss_map = None
        self._memory_refresh_enabled = False
        self._memory_refresh_interval_seconds = 0.0
        self._memory_refresh_next_at = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_rejoin_pending(self) -> bool:
        """Whether this runner already handed a deep link to Roblox/Bloxstrap.

        The dock watchdog has its own relaunch path. It must not start a second
        link while this runner is still waiting on the first handoff, even if
        Roblox temporarily has no discoverable window.
        """
        return self._rejoin_pending or self._rejoin_lock.locked()

    def claim_rejoin_launch(self) -> bool:
        """Reserve the single rejoin handoff for the dock watchdog.

        The watchdog cannot wait inside ``_attempt_rejoin`` because it also
        owns the loop that docks newly-created Roblox windows. It can claim
        the same pending state before launching, though, so the runner and
        watchdog cannot both open the deep link during a recovery race.
        """
        if not self._rejoin_lock.acquire(blocking=False):
            return False
        try:
            if self._rejoin_pending:
                return False
            self._rejoin_pending = True
            return True
        finally:
            self._rejoin_lock.release()

    def cancel_rejoin_launch(self) -> None:
        """Release a watchdog claim when its deep-link launch fails."""
        with self._rejoin_lock:
            self._rejoin_pending = False

    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def start(self, hwnd_getter, get_tasks, scroll_power: int = None, coords: dict = None,
              scroll_nudges: int = None, debug_screenshots: bool = False, default_walk_paths: dict = None,
              webhook: dict = None, expedition_color_buttons: bool = True,
              expedition_camera_o_ms: float = 100,
              memory_refresh_enabled: bool = False,
              memory_refresh_hours: float = MEMORY_REFRESH_DEFAULT_HOURS) -> dict:
        if self.is_running():
            return {"ok": False, "reason": "already_running"}
        self._stop_event = threading.Event()
        self._pause_event.clear()
        self._paused_logged = False
        self._stop_logged = False
        self._debug_screenshots = bool(debug_screenshots)
        self._expedition_color_buttons = bool(expedition_color_buttons)
        try:
            self._expedition_camera_o_ms = max(0.0, float(expedition_camera_o_ms))
        except (TypeError, ValueError):
            self._expedition_camera_o_ms = 100.0
        self._memory_refresh_enabled = bool(memory_refresh_enabled)
        try:
            refresh_hours = float(memory_refresh_hours)
        except (TypeError, ValueError):
            refresh_hours = MEMORY_REFRESH_DEFAULT_HOURS
        refresh_hours = min(MEMORY_REFRESH_MAX_HOURS,
                            max(MEMORY_REFRESH_MIN_HOURS, refresh_hours))
        self._memory_refresh_interval_seconds = refresh_hours * 3600.0
        self._memory_refresh_next_at = (
            time.monotonic() + self._memory_refresh_interval_seconds
            if self._memory_refresh_enabled else None)
        self._current_hwnd = None
        self._last_applied_team_loadout = None
        self._consecutive_losses = 0
        self._consecutive_loss_map = None
        self._thread = threading.Thread(
            target=self._run_session,
            args=(hwnd_getter, get_tasks, self._stop_event, scroll_power, coords, scroll_nudges,
                  default_walk_paths, webhook),
            daemon=True)
        self._thread.start()
        return {"ok": True}

    def start_debug_test(self, hwnd_getter, mode: str, macro_name: str, debug_screenshots: bool = False,
                           coords: dict = None, team_num=None) -> dict:
        """Run a selected Macro Operation test against the current Roblox
        screen, without lobby, map, stage, or battle setup. Goes through the
        exact same self._thread/self._stop_event start() itself uses (just
        targeting _run_debug_test instead of _run) specifically so this
        shows up as "running" the same way a real run does and F2/Stop
        actually interrupts it, instead of being a one-shot blocking call
        nothing can cancel (see debug_check_expedition_wave/
        debug_force_rejoin, which are that simpler kind on purpose -- this
        one needs the real thing)."""
        if self.is_running():
            return {"ok": False, "reason": "already_running"}
        if mode not in ("prestart", "battle", "team_loadout"):
            return {"ok": False, "reason": "bad_mode"}
        if mode != "team_loadout" and not macro_name:
            return {"ok": False, "reason": "no_macro"}
        if mode == "team_loadout":
            try:
                team_num = int(team_num)
            except (TypeError, ValueError):
                return {"ok": False, "reason": "bad_team"}
            if not 1 <= team_num <= TEAM_LOADOUT_MAX_SUPPORTED:
                return {"ok": False, "reason": "bad_team"}
        self._stop_event = threading.Event()
        self._pause_event.clear()
        self._paused_logged = False
        self._stop_logged = False
        self._debug_screenshots = bool(debug_screenshots)
        # Same saved click-point overrides a real run gets (see _run) --
        # without this, a debug test of blocks that click through
        # unit-info resets etc. would use the defaults the user may have
        # specifically re-picked away from.
        self._coords = {**DEFAULT_COORDS, **(coords or {})}
        self._current_hwnd = None
        self._last_applied_team_loadout = None
        self._memory_refresh_enabled = False
        self._memory_refresh_interval_seconds = 0.0
        self._memory_refresh_next_at = None
        self._thread = threading.Thread(
            target=self._run_debug_test,
            args=(hwnd_getter, mode, macro_name, team_num, self._stop_event),
            daemon=True)
        self._thread.start()
        return {"ok": True}

    def _run_debug_test(self, hwnd_getter, mode: str, macro_name: str, team_num,
                        stop_event: threading.Event) -> None:
        hwnd = hwnd_getter() if hwnd_getter else None
        if not hwnd or not wm.is_window(hwnd):
            self._log("[Debug] No Roblox window found -- can't test.")
            self._set_status(action="Idle")
            return
        self._current_hwnd = hwnd
        # A click/keypress has to actually reach the game, not whatever else
        # happened to have focus (this is fired from a Settings button click,
        # not guaranteed to be Roblox already) -- same focus-fix _run() does
        # before a real Start.
        wm.show_window(hwnd)
        if not wm.activate_window(hwnd):
            self._log("[Debug] Couldn't confirm Roblox actually took focus -- clicks may not register "
                       "until it does. Continuing anyway.")
        # Enough of a task shape for the selected test operation
        # to work with -- both only ever read task["macro"] and (for
        # _strip_auto_upgrade_for_expedition) task["mode"]/task["map"], not
        # anything about a real match in progress.
        task = {"macro": macro_name, "mode": "story", "map": "-", "difficulty": "-"}
        try:
            wm.prevent_sleep()  # same keep-awake as a real run (see _run_session)
            if mode == "team_loadout":
                self._log(f'[Debug] Testing Team Loadout {team_num}...')
                if not self._apply_team_loadout(hwnd, stop_event, task, team_override=team_num):
                    if not stop_event.is_set():
                        self._log("[Debug] Team Loadout test failed.")
            elif mode == "prestart":
                self._log(f'[Debug] Testing Pre Start blocks from "{macro_name}"...')
                self._run_prestart_blocks(hwnd, stop_event, task, first_repeat=True)
            else:
                self._log(f'[Debug] Testing Battle blocks from "{macro_name}"...')
                battle_blocks = self._load_battle_blocks(task)
                if not battle_blocks:
                    self._log(f'[Debug] Template "{macro_name}" has no Battle blocks.')
                else:
                    self._battle_block_index = 0
                    self._battle_block_state = {}
                    self._last_unit_ordinal = 0
                    self._quick_place_shift_down = False
                    # Ticks continuously (same as a real match's own poll
                    # loop) instead of running through the block list once
                    # and stopping -- Battle blocks are built to keep
                    # running for the length of a match (Upgrade Unit
                    # retrying over time, etc.), so this needs to keep
                    # ticking until Stop is pressed to actually exercise
                    # that, not just fire once.
                    while not self._checkpoint(stop_event):
                        self._run_battle_blocks_tick(hwnd, stop_event, battle_blocks, first_repeat=True,
                                                       macro_name=macro_name)
                        time.sleep(MATCH_RESULT_POLL_INTERVAL)
        finally:
            wm.allow_sleep()
            # Release this thread's screen-capture handle (see core.mss_manager)
            # when the debug test ends -- same cleanup a real run does below.
            vision.close_mss()
            if not (stop_event is not None and stop_event.is_set()):
                self._log("[Debug] Test finished.")
            self._set_status(action="Idle")

    def stop(self) -> dict:
        # Setting the event is enough on its own -- _checkpoint (called
        # between every step in _run) picks it up on the very next check,
        # and it also breaks any in-progress wait_for_image poll immediately
        # rather than letting that poll run its full timeout out. Clearing
        # pause here too so Stop always actually stops instead of leaving a
        # paused thread parked forever waiting on a resume that isn't coming.
        if self._stop_event is not None:
            self._stop_event.set()
        self._pause_event.clear()
        return {"ok": True}

    def pause(self) -> dict:
        if self.is_running():
            self._pause_event.set()
        return {"ok": True}

    def resume(self) -> dict:
        self._pause_event.clear()
        return {"ok": True}

    def _checkpoint(self, stop_event: threading.Event) -> bool:
        """Call between every major step. Blocks here while paused (so Pause
        works everywhere a stop check already happens, for free), then
        reports whether the caller should bail out. Centralizing the
        stop-check here (instead of the same 4-line block repeated after
        every step) is what makes it trivial to keep it consistent."""
        while self._pause_event.is_set() and not stop_event.is_set():
            if not self._paused_logged:
                self._log("[Macro] Paused.")
                self._set_status(action="Paused")
                self._paused_logged = True
            time.sleep(0.15)
        if self._paused_logged and not self._pause_event.is_set():
            self._log("[Macro] Resumed.")
            self._paused_logged = False
        if stop_event.is_set():
            # Say WHAT was in flight when the stop landed, not just
            # "Stopped." -- someone stopping a run that's visibly hung
            # (e.g. sitting on "Waiting for gamemode menu...") is exactly
            # the person who needs to know which step/image it was stuck
            # on, and the timeout message that would have named it never
            # gets to fire once they stop first. The action text is
            # whatever _set_status last showed on the Dashboard. Logged
            # once per stop (_stop_logged): every layer of the run calls
            # _checkpoint on the way out, which used to print a bare
            # "Stopped." per layer.
            if not self._stop_logged:
                self._stop_logged = True
                was = self._last_action
                if was and was not in ("Idle", "Paused"):
                    self._log(f"[Macro] Stopped. (was: {was})")
                else:
                    self._log("[Macro] Stopped.")
            self._set_status(action="Idle")
            return True
        return False

    def _interruptible_sleep(self, seconds: float, stop_event: threading.Event = None) -> None:
        """time.sleep(), but bails out immediately once stop_event fires
        instead of blocking it for the full duration -- F2/Stop is supposed
        to stay instant (see _checkpoint), which a plain time.sleep(5.0)
        settle delay quietly breaks for however long is left on it."""
        if seconds <= 0:
            return
        if stop_event is not None:
            stop_event.wait(seconds)
        else:
            time.sleep(seconds)

    def _click_return_to_lobby_if_found(self, hwnd, stop_event: threading.Event = None) -> bool:
        # Leave Stage can bring up its own "Return to Lobby" confirmation
        # (return.png) rather than backing out on its own -- optional/
        # best-effort like nav_disband and friends, so a real miss (it never
        # shows) is cheap. But a single instant find_image right after the
        # Leave Stage click was firing before the confirmation had even
        # animated in, so a real popup was being missed too -- reported as
        # "clicks Leave Stage, then just sits there" even though Return to
        # Lobby was genuinely up on screen a moment later. Short poll
        # instead of one-shot fixes that without meaningfully slowing down
        # the common case where it never appears.
        try:
            match = vision.wait_for_image(
                hwnd, "return", timeout=RETURN_TO_LOBBY_CHECK_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound:
            return False
        if match is None:
            return False

        for attempt in range(1, RETURN_TO_LOBBY_CLICK_RETRY_ATTEMPTS + 1):
            debug_path = self._debug_save(hwnd, "return", match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            retry = (f" (attempt {attempt}/{RETURN_TO_LOBBY_CLICK_RETRY_ATTEMPTS})"
                     if attempt > 1 else "")
            self._log(f"[Macro] Found \"Return to Lobby\" (score {match['score']:.2f})"
                      f"{retry} -- clicking it.{suffix}")
            if not wm.activate_window(hwnd):
                self._log("[Macro] Couldn't confirm focus before clicking \"Return to Lobby\" "
                          "-- click may not register.")
            vision.click_match(self._mouse, hwnd, match)
            self._interruptible_sleep(RETURN_TO_LOBBY_VERIFY_SETTLE, stop_event)
            if stop_event is not None and stop_event.is_set():
                return True

            try:
                still_there = vision.find_image(hwnd, "return")
            except vision.TemplateNotFound:
                still_there = None
            if still_there is None:
                return True
            match = still_there

        self._log(f'[Macro] "Return to Lobby" still showing after '
                  f'{RETURN_TO_LOBBY_CLICK_RETRY_ATTEMPTS} clicks -- continuing anyway.')
        return True

    def _click_close_popup_if_found(self, hwnd) -> None:
        # Spirit City Act 3 (Raid) can throw up a "Click anywhere to close"
        # popup (a boss/cutscene intro) mid-battle -- one-shot/best-effort
        # like nav_disband, checked every poll tick while watching for the
        # match result (see watch_close_popup in _wait_for_match_result).
        # Its visual variants all live in Assets/ui/click_anywhere_to_close/
        # and are tried automatically per search (see
        # vision.template_variant_paths).
        try:
            match = vision.find_image(hwnd, "click_anywhere_to_close")
        except vision.TemplateNotFound:
            return
        if match is None:
            return
        debug_path = self._debug_save(hwnd, "click_anywhere_to_close", match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f"[Macro] Found \"Click anywhere to close\" (score {match['score']:.2f}) -- clicking it.{suffix}")
        vision.click_match(self._mouse, hwnd, match)

    def _dismiss_reward_card_if_found(self, hwnd) -> bool:
        """A level-up "Select an upgrade!" reward-card modal can show up at
        several different moments in Expedition -- mid-battle right on top
        of the exp_extract choice, or (confirmed from a real stuck report)
        immediately after Victory, before the Repeat/Leave Stage result
        panel has even rendered, blocking repeat_stage from ever matching
        for the ENTIRE search window. Middle-screen click picks whatever
        card is there. Returns whether one was actually found (so callers
        can loop until it's actually gone, not just fire once)."""
        try:
            match = vision.find_image(hwnd, "select upgrade card")
        except vision.TemplateNotFound:
            return False
        if match is None:
            return False
        debug_path = self._debug_save(hwnd, "select upgrade card", match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f'[Macro] Found "select upgrade card" (score {match["score"]:.2f}) -- clicking it.{suffix}')
        # Cards are handed out for kills, so one cannot appear before the
        # fighting has started -- this is what Wait for Wave falls back to on
        # a gamemode with no wave counter.
        self._last_reward_card_at = time.time()
        self._last_board_disruption_at = self._last_reward_card_at
        task = getattr(self, "_active_task", None) or getattr(self, "_current_task", None) or {}
        if task.get("mode") == "story" and task.get("story_event") == "Eclipsed Infinite":
            preferred = task.get("eclipse_soul") or "Redeemed Soul"
            coord_keys = ECLIPSE_CARD_COORDS.get(preferred)
            if coord_keys:
                card_x, card_y = vision.ref_to_screen(
                    hwnd,
                    self._coords[coord_keys[0]],
                    self._coords[coord_keys[1]],
                )
                self._log(
                    f'[Macro] Eclipse Pick Card detected -- selecting configured '
                    f'"{preferred}" card at ({self._coords[coord_keys[0]]}, {self._coords[coord_keys[1]]}).'
                )
                self._mouse.click(card_x, card_y)
                return True
        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        self._mouse.click(left + self._coords["screen_middle_x"], top + self._coords["screen_middle_y"])
        return True

    def _clear_result_obtainment_modal(self, hwnd, stop_event: threading.Event = None) -> bool:
        """Dismiss a character/reward Obtainments modal over Victory.

        The modal can be opened by a stray click on the result panel's unit
        portraits. Its Close button disappears over another clickable result
        control, so use a zero-hold click and immediately park away to avoid
        clicking through into the newly exposed panel.
        """
        try:
            match = vision.find_image(
                hwnd, "result_modal_close", region=RESULT_MODAL_CLOSE_REGION)
        except vision.TemplateNotFound:
            return True
        if match is None:
            return True

        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        park_x, park_y = self._cxy("unit_info_reset")
        for attempt in range(1, 3):
            debug_path = self._debug_save(hwnd, "result_modal_close", match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f"[Macro] Found a Victory Obtainments modal (score {match['score']:.2f}) -- "
                       f"closing it (attempt {attempt}/2).{suffix}")
            if not wm.activate_window(hwnd):
                self._log("[Macro] Couldn't confirm focus before closing the Victory modal.")
            click_x, click_y = vision.ref_to_screen(hwnd, match["cx"], match["cy"])
            self._mouse.click(click_x, click_y, hold=0.0)
            self._mouse.move_to(left + park_x, top + park_y)
            if stop_event is not None and stop_event.is_set():
                return False
            time.sleep(0.3)
            try:
                match = vision.find_image(
                    hwnd, "result_modal_close", region=RESULT_MODAL_CLOSE_REGION)
            except vision.TemplateNotFound:
                return True
            if match is None:
                return True

        self._log("[Macro] Victory Obtainments modal still showing after 2 close attempts -- stopping.")
        return False


    def _debug_save(self, hwnd, name: str, match: dict) -> str:
        """Gated behind Settings > Debug > "Debug Match Screenshots" (off by
        default): a full-window screenshot on every single match found while
        running -- Play, Story, every stage row, ... -- adds up fast, so this
        only actually writes when the toggle is on. Returns None otherwise,
        which callers fold into their log line's " Debug: ..." suffix."""
        return vision.save_match_debug(
            hwnd, name, match, log=self._log) if self._debug_screenshots else None

    def _run_session(self, *args) -> None:
        """Thread entry for a real run. Wraps _run only to guarantee this
        thread's screen-capture handle (see core.mss_manager) is released when
        the run ends -- however it ends -- without threading a try/finally
        through _run's whole early-returning body."""
        try:
            # Keep the machine awake for the whole run. On macOS this is
            # essential: synthetic input doesn't reset the idle timer, so the
            # Mac sleeps mid-run despite the macro clicking (see
            # window_mac.prevent_sleep). Called here, on the session thread
            # that lives for the length of the run, because the Windows
            # backing (SetThreadExecutionState) is per-thread.
            wm.prevent_sleep()
            self._run(*args)
        except Exception as exc:
            # A daemon-thread exception is otherwise invisible in the
            # pythonw release build: Start simply becomes available again
            # and debug.log ends at the last normal action. Keep this final
            # boundary even though the independently recoverable phases in
            # _run are guarded below.
            self._log_unexpected_phase_error("runner session", exc)
            self._set_status(action="Idle")
        finally:
            wm.allow_sleep()
            vision.close_mss()

    def _log_unexpected_phase_error(self, phase: str, exc: Exception) -> None:
        self._log(
            f"[Macro] Unexpected error during {phase}: "
            f"{type(exc).__name__}: {exc}")
        self._log(f"[Debug] {traceback.format_exc().strip()}")

    def _run_guarded_phase(self, phase: str, hwnd, stop_event: threading.Event,
                           operation):
        """Run one independently recoverable part of an unattended pass.

        Returns ``(completed, result)``. An implementation bug or transient
        capture exception in Auto Bounty/Challenge/a single queued task must
        not kill the daemon thread and silently end the entire overnight run.
        """
        try:
            return True, operation()
        except Exception as exc:
            self._log_unexpected_phase_error(phase, exc)
            if not stop_event.is_set():
                self._set_status(action=f"Recovering after {phase} error...")
                try:
                    self._recover_to_lobby(hwnd, stop_event)
                except Exception as recovery_exc:
                    self._log_unexpected_phase_error(
                        f"{phase} lobby recovery", recovery_exc)
            return False, None

    def _run_crafting_if_due(self, hwnd, stop_event: threading.Event) -> None:
        if self._crafting_wants_in():
            self._run_crafting(hwnd, stop_event)

    def _run_fuel_refill_if_due(self, hwnd, stop_event: threading.Event) -> None:
        if self._fuel_wants_in():
            self._run_fuel_refill(hwnd, stop_event)

    def _memory_refresh_due(self) -> bool:
        """Return whether the optional long-run Roblox refresh is due.

        This is intentionally a cheap monotonic-clock check. It is called
        only after a match result, alongside the existing safe-boundary
        schedulers; there is no independent timer thread touching capture or
        input while gameplay is active.
        """
        return bool(
            self._memory_refresh_enabled
            and self._memory_refresh_interval_seconds > 0
            and self._memory_refresh_next_at is not None
            and time.monotonic() >= self._memory_refresh_next_at
        )

    def _complete_memory_refresh(self) -> None:
        """Arm the next refresh after a successful Roblox rejoin."""
        self._memory_refresh_next_at = (
            time.monotonic() + self._memory_refresh_interval_seconds)

    def _run(self, hwnd_getter, get_tasks, stop_event: threading.Event, scroll_power: int = None,
              coords: dict = None, scroll_nudges: int = None, default_walk_paths: dict = None,
              webhook: dict = None) -> None:
        coords = {**DEFAULT_COORDS, **(coords or {})}
        # Also kept on self: most click sites live in methods coords was
        # never threaded through -- one shared dict beats adding a parameter
        # to a dozen call chains (see _cxy).
        self._coords = coords
        default_walk_paths = default_walk_paths or {}
        webhook = webhook or {}
        self._active_task_progress = None
        self._current_task = None

        hwnd = hwnd_getter()
        if not hwnd or not wm.is_window(hwnd):
            self._log("[Macro] Roblox isn't docked yet -- can't start.")
            self._set_status(action="Idle")
            return
        self._current_hwnd = hwnd
        # Kept for _attempt_rejoin -- after a rejoin relaunches Roblox, the
        # dock watchdog (main.py) re-docks it under a NEW hwnd on its own;
        # this is how the runner finds out what that new hwnd actually is.
        self._hwnd_getter = hwnd_getter

        # A click has to actually reach the game, not this panel -- same
        # focus-fix every other live-input action in this app uses.
        wm.show_window(hwnd)
        if not wm.activate_window(hwnd):
            self._log("[Macro] Couldn't confirm Roblox actually took focus -- clicks may not register "
                       "until it does. Continuing anyway.")

        # SendInput cannot reach a window owned by a HIGHER-privilege
        # process than this one -- Windows drops it with no error at all,
        # which looks exactly like "finds the button, clicks it, nothing
        # happens, cursor doesn't even move" (a persistent user report even
        # after several unrelated click-reliability fixes). Checked once
        # here rather than every click since elevation doesn't change
        # mid-run, and this is advisory only -- it can't fix a mismatch,
        # only explain one if it exists.
        if wm.is_process_elevated(hwnd) and not wm.is_self_elevated():
            self._log("[Macro] Roblox appears to be running as Administrator, but this macro isn't -- "
                       "Windows silently blocks simulated clicks/keys from a non-elevated app to an "
                       "elevated one. If clicks aren't registering, close both and relaunch this macro "
                       "as Administrator too (right-click > Run as administrator).")

        # Already standing in a match? nav_unitmanager only renders once
        # you're actually in-game (it's the same image _wait_teleport_in
        # treats as the "we teleported in" confirmation), so seeing it now
        # means the user pressed Start from INSIDE a stage -- all the lobby
        # navigation (Play, gamemode, map, stage, matchmaking, teleport)
        # would be clicking at screens that aren't there. The first task
        # skips straight to Pre Start instead (see _run_task's
        # _skip_first_task_setup), picking the run up from where the user
        # already is.
        self._skip_first_task_setup = False
        try:
            if vision.find_image(hwnd, "nav_unitmanager") is not None:
                self._skip_first_task_setup = True
                self._log("[Macro] Unit Manager is visible -- already in-game, so the first task "
                           "skips stage entry and starts from Pre Start.")
        except vision.TemplateNotFound:
            pass

        # Bounty and Challenge run ONCE per Start (not once per task-queue
        # pass, see
        # the while loop below) -- if it's enabled, every ready stage slot
        # gets attempted before the Task Queue ever starts. Skipped when
        # starting already in-game: its navigation begins at the lobby,
        # which is exactly where we aren't -- ready slots still get their
        # chance at the between-repeats check once the current stage ends.
        if self._checkpoint(stop_event):
            return
        if not self._skip_first_task_setup:
            bounty_ok, bounty_result = self._run_guarded_phase(
                "Auto Bounty", hwnd, stop_event,
                lambda: self._run_bounties(
                    hwnd, stop_event, coords, default_walk_paths, webhook))
            bounty_enabled = bool(bounty_result) if bounty_ok else bool(
                self._bounty_settings().get("enabled"))
        else:
            bounty_enabled = False
        if self._checkpoint(stop_event):
            return
        if not self._skip_first_task_setup:
            self._run_guarded_phase(
                "Challenge", hwnd, stop_event,
                lambda: self._run_challenges(
                    hwnd, stop_event, coords, default_walk_paths, webhook))
            self._run_guarded_phase(
                "Auto Crafting", hwnd, stop_event,
                lambda: self._run_crafting_if_due(hwnd, stop_event))
            self._run_guarded_phase(
                "Auto Fuel", hwnd, stop_event,
                lambda: self._run_fuel_refill_if_due(hwnd, stop_event))
            shop_enabled = self._auto_shop_wants_in()
            self._run_guarded_phase(
                "Auto Shop", hwnd, stop_event,
                lambda: self._run_auto_shop_if_due(hwnd, stop_event))
        if self._checkpoint(stop_event):
            return

        loop_pass = 1
        while True:
            # Re-read the queue every pass instead of once up front -- a run
            # loops indefinitely (see below), potentially for hours, and the
            # user may edit the Task screen (add/remove/reorder) between
            # passes expecting the NEXT pass to pick up their changes rather
            # than keep replaying a stale snapshot from when the run started.
            tasks = get_tasks()
            if not tasks:
                if bounty_enabled:
                    self._log("[Macro] Auto Bounty pass finished and the Task Queue is empty -- going Idle.")
                elif shop_enabled:
                    self._log("[Macro] Auto Shop pass finished and the Task Queue is empty -- going Idle.")
                else:
                    self._log("[Macro] Task queue is empty -- add a task on the Task screen first.")
                self._set_status(action="Idle")
                return

            if loop_pass == 1:
                self._log(f"[Macro] Starting run -- {len(tasks)} task(s) queued.")
            else:
                self._log(f"[Macro] Task queue finished -- restarting from task 1 (pass {loop_pass}).")

            for task_index, task in enumerate(tasks, start=1):
                if self._checkpoint(stop_event):
                    return

                map_name = task.get("map")
                # Event mode has no map to pick (just an Act) -- it's the one
                # mode where a missing map is expected, not a misconfigured
                # task, so don't skip it over that.
                if not map_name and (task.get("mode") or "story") != "event":
                    self._log(f"[Macro] Task {task_index}/{len(tasks)} has no map set -- skipping it.")
                    continue

                # A disconnect/rejoin during a previous task (see
                # _attempt_rejoin) may have re-docked Roblox under a new
                # hwnd -- pick that up before starting the next task.
                if self._current_hwnd and wm.is_window(self._current_hwnd):
                    hwnd = self._current_hwnd

                # A mid-task failure (a stuck battle, a missed click, ...)
                # doesn't kill the whole overnight run -- _run_task recovers to
                # the lobby and retries internally, only returning False when
                # stop_event actually fired or a non-retriable configuration
                # error (such as a missing Boss Rush route) requires stopping.
                task_ok, task_result = self._run_guarded_phase(
                    f"task {task_index}/{len(tasks)}", hwnd, stop_event,
                    lambda: self._run_task(
                        hwnd, stop_event, task, task_index, len(tasks), coords,
                        scroll_power, scroll_nudges, default_walk_paths, webhook))
                if not task_ok:
                    if self._checkpoint(stop_event):
                        return
                    # Recovery already returned to the lobby. Skip only this
                    # broken task and let the remaining queue (or its next
                    # pass) continue instead of ending the runner thread.
                    continue
                if not task_result:
                    self._set_status(action="Idle")
                    return

                # Auto Challenge after each finished task (we're back at the
                # lobby here). This is what catches the cases the between-
                # repeats hook inside _run_task can't: a single-repeat task,
                # or the LAST repeat of any task (the in-task hook is gated
                # by `not is_last_repeat`). Without this, a slot that became
                # ready at the last :00/:30 boundary just sat unplayed until
                # the next Start press -- the "challenges are ready but it
                # keeps doing the tasks normally" report. Runs before the
                # other diversions because Challenge has priority when
                # several come due at once (same ordering as the start of
                # _run and the between-repeats hook).
                if self._challenge_has_ready_stage():
                    self._run_guarded_phase(
                        "Challenge", hwnd, stop_event,
                        lambda: self._run_challenges(
                            hwnd, stop_event, coords, default_walk_paths, webhook))
                if self._current_hwnd and wm.is_window(self._current_hwnd):
                    hwnd = self._current_hwnd
                if self._checkpoint(stop_event):
                    return

                # Auto Crafting after each finished task (we're back at the
                # lobby here). This is what catches the cases the between-
                # repeats hook inside _run_task can't: a single-repeat task, or
                # the win threshold being reached on a task's LAST repeat (the
                # in-task hook is gated by `not is_last_repeat`). Without this,
                # a reached threshold just sat until the next Start -- the exact
                # "task finished, came back to lobby, no craft" report.
                self._run_guarded_phase(
                    "Auto Crafting", hwnd, stop_event,
                    lambda: self._run_crafting_if_due(hwnd, stop_event))
                if self._current_hwnd and wm.is_window(self._current_hwnd):
                    hwnd = self._current_hwnd
                if self._checkpoint(stop_event):
                    return
                self._run_guarded_phase(
                    "Auto Fuel", hwnd, stop_event,
                    lambda: self._run_fuel_refill_if_due(hwnd, stop_event))
                if self._current_hwnd and wm.is_window(self._current_hwnd):
                    hwnd = self._current_hwnd
                if self._checkpoint(stop_event):
                    return
                self._run_guarded_phase(
                    "Auto Shop", hwnd, stop_event,
                    lambda: self._run_auto_shop_if_due(hwnd, stop_event))
                if self._current_hwnd and wm.is_window(self._current_hwnd):
                    hwnd = self._current_hwnd
                if self._checkpoint(stop_event):
                    return

            # The queue always loops back to task 1 once it finishes rather
            # than going Idle -- Stop (F2) is the only way to actually end
            # an unattended run now.
            if self._checkpoint(stop_event):
                return
            loop_pass += 1





    def _run_task(self, hwnd, stop_event: threading.Event, task: dict, task_index: int, task_count: int,
                   coords: dict, scroll_power: int, scroll_nudges: int, default_walk_paths: dict,
                   webhook: dict) -> bool:
        """Runs one task's full repeat count end to end. A mid-task failure
        backs out to the lobby and retries this task's setup from scratch
        (see _recover_to_lobby), up to TASK_RECOVERY_ATTEMPTS times, before
        giving up on just this task and letting the run move on to the next
        one -- so one stuck battle doesn't end an unattended overnight run.
        Returns False when stop_event fired or a non-retriable configuration
        error requires stopping the runner; True in every other case,
        including "gave up on this task after repeated failures".
        """
        mode = task.get("mode") or "story"
        # The Task Builder exposes the future-proof Event schema, while the
        # existing Summer runner already owns the navigation and portal
        # handling. Normalize the current Event implementation at the runner
        # boundary so all existing Summer retry/result paths remain shared.
        if mode == "event":
            task = dict(task)
            task["mode"] = "summer"
            task["summer_mode"] = task.get("event_gamemode") or "Event Mode"
            if task["summer_mode"] == "Portal Mode":
                task["map"] = task.get("map") or "Summer Portal"
            else:
                task["map"] = "Summer"
            mode = "summer"
        if mode == "boss_rush":
            boss_rush_state = self._new_boss_rush_state(task)
            if boss_rush_state is None:
                self._log("[Boss Rush] Invalid gate count or card preference. Set gates to 1-6 and choose left, middle, or right.")
                return False
        map_name = task.get("map")
        repeat_total = max(1, int(task.get("repeat") or 1))
        progress_task = dict(task)
        progress_task["map"] = map_name or mode.title()
        # The running task, so mid-match handlers can ask which map they are
        # on without the task being threaded through every tick.
        self._current_task = task
        self._active_task_progress = {
            "index": task_index,
            "count": task_count,
            "map": progress_task["map"],
            "next_repeat": 1,
            "repeat_total": repeat_total,
        }
        task_started_at = time.monotonic()
        self._send_progress_webhook(
            webhook,
            progress_task,
            f"Task {task_index}/{task_count} Started",
            f'Starting **{progress_task["map"]}** x{repeat_total}.',
            0x5865F2,
            extra_fields=[
                {"name": "Mode", "value": mode.title(), "inline": True},
                {"name": "Repeats", "value": str(repeat_total), "inline": True},
            ],
            current_action=f"Task {task_index}/{task_count} -- starting",
            next_phase=f"Enter stage, then Pre Start (repeat 1/{repeat_total})",
        )

        resume_repeat = 1
        portal_task = (
            (mode == "summer" and task.get("summer_mode") == "Portal Mode")
            or (mode == "event" and task.get("event_gamemode") == "Portal Mode")
        )
        recovery_attempt = 0
        while recovery_attempt < TASK_RECOVERY_ATTEMPTS or portal_task:
            recovery_attempt += 1
            if self._checkpoint(stop_event):
                return False
            # A disconnect handled during the previous attempt (see
            # _attempt_rejoin) may have re-docked Roblox under a NEW hwnd --
            # self._current_hwnd is what tracks that, so every retry picks
            # up wherever the game actually ended up rather than continuing
            # to act on a hwnd that might already be dead.
            if self._current_hwnd and wm.is_window(self._current_hwnd):
                hwnd = self._current_hwnd
            if recovery_attempt > 1:
                attempt_label = (
                    f"{recovery_attempt}/continuous"
                    if portal_task else f"{recovery_attempt}/{TASK_RECOVERY_ATTEMPTS}"
                )
                self._log(f'[Macro] Retrying task {task_index}/{task_count} from the lobby '
                           f'(attempt {attempt_label})...')
            self._log(f'[Macro] Task {task_index}/{task_count}: "{map_name}" x{repeat_total}.')
            # Everything beyond current_task/current_repeat/map/action is
            # new -- for the Dashboard's Status Readout hover-expand (shows
            # the fuller picture of what's actually running, not just the
            # always-visible summary row). mode/stage/difficulty don't
            # apply the same way to every mode (Expedition has no "stage",
            # Raid/Infinite/Mastery lock difficulty, Challenge has neither
            # in the Task Queue sense) -- placeholder "-" rather than
            # leaving a PREVIOUS task's value stale on screen.
            self._set_status(current_task=f"{task_index} / {task_count}",
                              current_repeat=f"{resume_repeat} / {repeat_total}",
                              map=map_name, action="Starting...", mode=mode, stage=str(task.get("stage") or "-"),
                              difficulty=task.get("difficulty") or "-", play_mode=task.get("play_mode") or "solo",
                              macro=task.get("macro") or "-")

            # Everything from the lobby through the first teleport-in runs
            # ONCE per task -- every repeat after that re-enters the same
            # stage directly via Repeat Stage (see _handle_match_result),
            # skipping the lobby/gamemode/map/stage picks entirely.
            # Consumed one-shot: the started-already-in-game case (see _run's
            # nav_unitmanager check) skips even that first entry -- the user
            # is standing in the stage right now, so the run picks up at Pre
            # Start. Any retry/recovery after that goes through the real
            # setup, since a recovery lands back in the lobby.
            skip_setup = getattr(self, "_skip_first_task_setup", False)
            self._skip_first_task_setup = False
            if skip_setup:
                self._log(f'[Macro] Already in-game -- treating the current stage as "{map_name}" '
                           f'and skipping stage entry.')
            elif not self._run_task_setup(hwnd, stop_event, task, mode, map_name, coords, scroll_power,
                                            scroll_nudges, webhook):
                if stop_event.is_set():
                    return False
                if (mode == "summer" and task.get("summer_mode") == "Portal Mode"):
                    # Portal setup is intentionally one-shot. Retrying a
                    # failed start from the lobby would run nav_event again
                    # only if the setup itself succeeded and the failure was
                    # inside the match. If setup fails, recover to a known
                    # lobby and retry the unfinished portal task instead of
                    # stopping the entire macro.
                    self._log(
                        "[Macro] Portal setup failed -- recovering to the lobby "
                        "and retrying the unfinished portal task."
                    )
                    if stop_event.is_set():
                        return False
                    self._recover_to_lobby(hwnd, stop_event)
                    continue
                if not self._recover_to_lobby(hwnd, stop_event):
                    return not stop_event.is_set()
                continue

            task_failed = False
            # True on the genuine first repeat, AND on any later repeat that
            # just went through a full fresh _run_task_setup re-entry
            # (matchmaking's own repeat re-entry, or resuming after a
            # Challenge interleave below) -- either way that's a real fresh
            # entry into the stage, needing Team Loadout and the Walk Path
            # block to run again exactly like the actual first repeat does,
            # not "repeat_index == 1" alone (which used to wrongly skip both
            # on every repeat past the first even when _run_task_setup had
            # just fully re-run, confirmed from a real report: Walk Path
            # silently skipped resuming a task after a Challenge interleave).
            fresh_entry = True
            for repeat_index in range(resume_repeat, repeat_total + 1):
                self._active_task_progress["next_repeat"] = repeat_index
                self._set_status(current_repeat=f"{repeat_index} / {repeat_total}")
                battle_started = time.time()
                result = self._play_one_match(hwnd, stop_event, task, default_walk_paths,
                                                first_repeat=fresh_entry, webhook=webhook)
                fresh_entry = False
                if result == "configuration_error":
                    self._set_status(action="Idle")
                    return False
                if result is None:
                    if stop_event.is_set():
                        return False
                    resume_repeat = repeat_index
                    task_failed = True
                    break
                duration = self._format_duration(time.time() - battle_started)
                # Both the Infinite wave-limit exit ("wave_limit") and a
                # "Leave at Minute" battle block ("left") leave the LIVE match
                # to the lobby themselves -- no Victory/Defeat screen follows,
                # and the next repeat must re-enter from the lobby.
                left_live_match = result in ("wave_limit", "left")

                # Consecutive-loss fail-safe: a genuine unbroken loss streak
                # on THIS map (not just losses somewhere in the run) usually
                # means something's actually wrong -- a bad loadout, a stuck
                # client, a map that's too hard -- rather than plain bad
                # luck, so it's worth a full Roblox restart instead of just
                # keep feeding it more attempts. Any win, or a loss on a
                # DIFFERENT map (Challenge/task interleaving can switch maps
                # between repeats), resets the count.
                if result == "loss" and self._consecutive_loss_map == map_name:
                    self._consecutive_losses += 1
                elif result == "loss":
                    self._consecutive_loss_map = map_name
                    self._consecutive_losses = 1
                else:
                    self._consecutive_losses = 0
                    self._consecutive_loss_map = None
                restart_needed = self._consecutive_losses >= MAX_CONSECUTIVE_LOSSES_SAME_MAP

                is_last_repeat = repeat_index == repeat_total
                # Challenge used to only ever get checked once, right at the
                # very start of a Start press, before the Task Queue even
                # began -- a task with a huge repeat count (effectively
                # "forever") meant Challenge's own cooldown resetting
                # partway through a run was never revisited at all. Checked
                # here too now, between every repeat: if a slot's ready,
                # this repeat forces a real Leave Stage (same as the task's
                # own last repeat would) instead of Repeat Stage, so the
                # task cleanly steps out of its own stage before Challenge's
                # navigation (which starts from the lobby) runs.
                challenge_wants_in = (not is_last_repeat) and self._challenge_has_ready_stage()
                # Same interleave shape for Auto Crafting: if the win counter
                # reaches its threshold INCLUDING this result, force a real
                # Leave Stage now so crafting navigation can start from the
                # lobby. _handle_match_result persists this win immediately
                # afterward; the projection here only decides Repeat vs Leave.
                crafting_wants_in = (
                    (not is_last_repeat)
                    and self._crafting_wants_in(task, result)
                )
                # Auto Fuel is checked at the same safe boundary. A due timer
                # forces Leave Stage now, then the pass runs from the lobby
                # before this same task is entered again.
                fuel_wants_in = (not is_last_repeat) and self._fuel_wants_in()
                # Auto Shop runs last among resource diversions. It uses the
                # same leave-to-lobby boundary and never interrupts a match.
                auto_shop_wants_in = (
                    (not is_last_repeat)
                    and self._auto_shop_wants_in()
                )
                # Periodic memory refresh uses the same safe-boundary shape
                # as the other maintenance diversions: leave after this
                # result, relaunch Roblox into the lobby, then re-enter the
                # task from scratch. Other scheduled diversions take priority
                # so two maintenance flows never stack on one result.
                memory_refresh_wants_in = (
                    self._memory_refresh_due()
                    and not restart_needed
                    and not challenge_wants_in
                    and not crafting_wants_in
                    and not fuel_wants_in
                    and not auto_shop_wants_in
                )
                # The bounded-Infinite path and the Leave-at-Minute block
                # (left_live_match) already left the live match, so there is no
                # Victory/Defeat screen to process here.
                handled_result = left_live_match or self._handle_match_result(
                    hwnd, stop_event, task, result, duration, webhook,
                    repeat=(not is_last_repeat) and not challenge_wants_in
                    and not crafting_wants_in and not fuel_wants_in
                    and not auto_shop_wants_in
                    and not restart_needed and not memory_refresh_wants_in)
                if (not handled_result and result == "win" and not is_last_repeat
                        and task.get("mode") == "summer"
                        and task.get("summer_mode") == "Portal Mode"):
                    # A dropped click can leave us on either the Victory
                    # screen or the already-open Portal Selection screen.
                    # Retry this in-game transition before allowing the normal
                    # task-recovery path to return through nav_event.
                    self._log("[Macro] Retrying the portal repeat flow without leaving the game.")
                    for portal_attempt in range(1, 4):
                        handled_result = self._select_summer_portal_for_repeat(
                            hwnd, stop_event, task)
                        if handled_result:
                            break
                        self._log(
                            f"[Macro] Portal repeat attempt {portal_attempt}/3 failed."
                        )
                        if stop_event.is_set():
                            break
                        self._interruptible_sleep(1.0, stop_event)
                if not handled_result:
                    if stop_event.is_set():
                        return False
                    task_failed = True
                    break
                if self._checkpoint(stop_event):
                    return False

                if restart_needed:
                    if portal_task:
                        self._log(f'[Macro] Lost {self._consecutive_losses}x in a row on '
                                  f'"{map_name}" -- three portal hearts used, stopping this task.')
                        task_failed = True
                        break
                    self._log(f'[Macro] Lost {self._consecutive_losses}x in a row on "{map_name}" -- '
                               f'restarting Roblox as a fail-safe.')
                    screenshot_path = self._save_debug_screenshot_unconditional(hwnd, "consecutive_loss_restart")
                    self._send_event_webhook(
                        webhook, task, "Consecutive Losses -- Restarting Roblox",
                        f'Lost {self._consecutive_losses}x in a row on "{map_name}" -- restarting Roblox '
                        f"before retrying.", 0xE05A6D, screenshot_path)
                    self._consecutive_losses = 0
                    self._consecutive_loss_map = None
                    if not self._attempt_rejoin(hwnd, stop_event):
                        if stop_event.is_set():
                            return False
                        task_failed = True
                        break
                    if self._current_hwnd and wm.is_window(self._current_hwnd):
                        hwnd = self._current_hwnd
                    if not is_last_repeat:
                        # Leave Stage above already left the stage entirely
                        # (repeat=False), and the restart just left the
                        # lobby fresh too -- re-enter the task from scratch
                        # exactly like the very first repeat did, same as
                        # the Challenge-interleave case right below.
                        if not self._run_task_setup(hwnd, stop_event, task, mode, map_name, coords,
                                                      scroll_power, scroll_nudges, webhook):
                            if stop_event.is_set():
                                return False
                            task_failed = True
                            break
                        fresh_entry = True
                    continue

                if challenge_wants_in:
                    self._active_task_progress["next_repeat"] = repeat_index + 1
                    self._log(f'[Macro] Challenge stage ready -- pausing "{map_name}" to run it '
                               f'before continuing.')
                    self._run_challenges(hwnd, stop_event, coords, default_walk_paths, webhook)
                    if self._checkpoint(stop_event):
                        return False
                    # Challenge has priority when both diversions become due
                    # on the same result. It can also add qualifying wins of
                    # its own, so honor a now-due crafting pass while already
                    # in the lobby instead of re-entering the farm for an
                    # unnecessary extra match first.
                    if self._crafting_wants_in():
                        self._log(
                            "[Macro] Crafting became due during Challenge -- "
                            "running it before resuming the farm.")
                        self._run_crafting(hwnd, stop_event)
                        if self._checkpoint(stop_event):
                            return False
                    self._log(f'[Macro] Challenge pass finished -- resuming "{map_name}".')
                    # Left the stage entirely for Challenge (repeat=False
                    # above already did Leave Stage + Return to Lobby), so
                    # this repeat re-enters the task from scratch exactly
                    # like the very first one did, not a quick Repeat Stage
                    # requeue -- there's no stage left to requeue INTO.
                    if not self._run_task_setup(hwnd, stop_event, task, mode, map_name, coords,
                                                  scroll_power, scroll_nudges, webhook):
                        if stop_event.is_set():
                            return False
                        task_failed = True
                        break
                    fresh_entry = True
                    continue

                if crafting_wants_in:
                    self._log(f'[Macro] Crafting due -- pausing "{map_name}" to run a crafting pass '
                               f'before continuing.')
                    self._run_crafting(hwnd, stop_event)
                    if self._checkpoint(stop_event):
                        return False
                    self._log(f'[Macro] Crafting pass finished -- resuming "{map_name}".')
                    # Same as the Challenge interleave above: the stage was left
                    # entirely (repeat=False), so re-enter from scratch, not a
                    # Repeat Stage requeue -- there's no stage left to requeue into.
                    if not self._run_task_setup(hwnd, stop_event, task, mode, map_name, coords,
                                                  scroll_power, scroll_nudges, webhook):
                        if stop_event.is_set():
                            return False
                        task_failed = True
                        break
                    fresh_entry = True
                    continue

                if fuel_wants_in:
                    self._log(f'[Macro] Auto Fuel is due. Pausing "{map_name}" at a safe boundary.')
                    self._run_fuel_refill(hwnd, stop_event)
                    if self._checkpoint(stop_event):
                        return False
                    if self._current_hwnd and wm.is_window(self._current_hwnd):
                        hwnd = self._current_hwnd
                    self._log(f'[Macro] Auto Fuel pass finished. Resuming "{map_name}".')
                    # Auto Fuel temporarily owns the Dashboard status context.
                    # Restore the same task and the upcoming repeat before
                    # re-entering so Current Task/Repeat never stay stale.
                    self._set_status(
                        current_task=f"{task_index} / {task_count}",
                        current_repeat=f"{repeat_index + 1} / {repeat_total}",
                        map=map_name,
                        action="Resuming after Auto Fuel...",
                        mode=mode,
                        stage=str(task.get("stage") or "-"),
                        difficulty=task.get("difficulty") or "-",
                        play_mode=task.get("play_mode") or "solo",
                        macro=task.get("macro") or "-",
                    )
                    if not self._run_task_setup(
                            hwnd, stop_event, task, mode, map_name, coords,
                            scroll_power, scroll_nudges, webhook):
                        if stop_event.is_set():
                            return False
                        task_failed = True
                        break
                    fresh_entry = True
                    continue

                if auto_shop_wants_in:
                    self._log(
                        f'[Macro] Auto Shop is due. Pausing "{map_name}" '
                        "at a safe boundary."
                    )
                    self._run_auto_shop(hwnd, stop_event)
                    if self._checkpoint(stop_event):
                        return False
                    if self._current_hwnd and wm.is_window(self._current_hwnd):
                        hwnd = self._current_hwnd
                    self._log(
                        f'[Macro] Auto Shop pass finished. Resuming "{map_name}".'
                    )
                    self._set_status(
                        current_task=f"{task_index} / {task_count}",
                        current_repeat=f"{repeat_index + 1} / {repeat_total}",
                        map=map_name,
                        action="Resuming after Auto Shop...",
                        mode=mode,
                        stage=str(task.get("stage") or "-"),
                        difficulty=task.get("difficulty") or "-",
                        play_mode=task.get("play_mode") or "solo",
                        macro=task.get("macro") or "-",
                    )
                    if not self._run_task_setup(
                            hwnd, stop_event, task, mode, map_name, coords,
                            scroll_power, scroll_nudges, webhook):
                        if stop_event.is_set():
                            return False
                        task_failed = True
                        break
                    fresh_entry = True
                    continue

                if memory_refresh_wants_in:
                    self._log(
                        "Periodic Roblox refresh is due -- waiting at the "
                        "safe lobby boundary before restarting the client.")
                    if not self._attempt_rejoin(hwnd, stop_event):
                        if stop_event.is_set():
                            return False
                        task_failed = True
                        break
                    self._complete_memory_refresh()
                    if self._current_hwnd and wm.is_window(self._current_hwnd):
                        hwnd = self._current_hwnd
                    if not is_last_repeat:
                        if not self._run_task_setup(
                                hwnd, stop_event, task, mode, map_name, coords,
                                scroll_power, scroll_nudges, webhook):
                            if stop_event.is_set():
                                return False
                            task_failed = True
                            break
                        fresh_entry = True
                    continue

                if not is_last_repeat:
                    if left_live_match or task.get("play_mode") == "matchmaking":
                        # Leave Stage (see _handle_match_result -- matchmaking
                        # always leaves, never Repeat Stage), or the Infinite
                        # wave-limit exit, puts us back in the lobby rather
                        # than a repeat teleport -- re-enter from scratch.
                        if not self._run_task_setup(hwnd, stop_event, task, mode, map_name, coords,
                                                      scroll_power, scroll_nudges, webhook):
                            if stop_event.is_set():
                                return False
                            task_failed = True
                            break
                        fresh_entry = True
                    elif not self._wait_teleport_in(hwnd, stop_event, webhook, task):
                        if stop_event.is_set():
                            return False
                        task_failed = True
                        break

            if not task_failed:
                elapsed = self._format_duration(time.monotonic() - task_started_at)
                self._send_progress_webhook(
                    webhook,
                    progress_task,
                    f"Task {task_index}/{task_count} Finished",
                    f'Task **{progress_task["map"]}** completed all {repeat_total} repeat(s).',
                    0x3FBF6F,
                    extra_fields=[
                        {"name": "Status", "value": "Completed", "inline": True},
                        {"name": "Elapsed", "value": elapsed, "inline": True},
                    ],
                    current_action=f"Task {task_index}/{task_count} -- completed",
                    next_phase=self._next_task_progress(task_index, task_count),
                )
                return True  # finished every repeat cleanly

            if (task.get("mode") == "summer"
                    and task.get("summer_mode") == "Portal Mode"):
                # A portal result can occasionally fail to render. After the
                # three bounded in-game attempts above, leave cleanly so the
                # queue can continue with its next task.
                self._log(
                    "[Macro] Portal flow failed 3 times -- returning to the lobby "
                    "and retrying the unfinished portal task."
                )
                resume_repeat = max(1, self._active_task_progress.get("next_repeat", 1))
                if not self._recover_to_lobby(hwnd, stop_event):
                    if stop_event.is_set():
                        return False
                    self._log("[Macro] Portal lobby recovery did not confirm yet -- retrying recovery.")
                # Do not report success: repeat_total has not been reached.
                # The next recovery attempt re-enters this same task and the
                # repeat loop starts again instead of advancing the queue.
                continue
            self._log(f'[Macro] Task {task_index}/{task_count} hit a problem mid-run -- recovering to the lobby.')
            if not self._recover_to_lobby(hwnd, stop_event):
                return not stop_event.is_set()  # couldn't even get back to the lobby -- nothing left to retry

        self._log(f'[Macro] Task {task_index}/{task_count} still failing after '
                   f'{TASK_RECOVERY_ATTEMPTS} attempts -- giving up on it.')
        screenshot_path = self._save_debug_screenshot_unconditional(hwnd, "task_gave_up")
        self._send_event_webhook(
            webhook, task, "Task Gave Up",
            f"Task {task_index}/{task_count} still failing after {TASK_RECOVERY_ATTEMPTS} recovery "
            f"attempts -- moving on to the next task.", 0xE05A6D, screenshot_path)
        return True

    def _recover_to_lobby(self, hwnd, stop_event: threading.Event) -> bool:
        """Best-effort recovery after a mid-task failure -- tries Leave
        Stage first (in case a match/Pre Start is still up), then spams
        Back (in case it's stuck on a menu instead), then confirms the
        lobby is actually reached. Returns whether it actually got there."""
        self._log("[Macro] Attempting to recover to the lobby...")
        self._set_status(action="Recovering...")
        if self._checkpoint(stop_event):
            return False
        try:
            leave_match = vision.find_image(hwnd, "leave_stage")
        except vision.TemplateNotFound:
            leave_match = None
        if leave_match is not None:
            self._log(f"[Macro] Found Leave Stage (score {leave_match['score']:.2f}) -- clicking it.")
            vision.click_match(self._mouse, hwnd, leave_match)
            self._interruptible_sleep(SETTLE_DELAY, stop_event)
            self._click_return_to_lobby_if_found(hwnd, stop_event)
        if self._checkpoint(stop_event):
            return False
        self._spam_back_until_gone(hwnd, stop_event)
        if self._checkpoint(stop_event):
            return False
        return self._ensure_lobby(hwnd, stop_event)

    def _handle_structured_failure(self, report: FailureReport, hwnd=None) -> bool:
        """Handles structured failure reports to enforce bounded recovery logic."""
        self._log(f"[Diagnostics] [{report.category.name}] {report.user_message} (Action: {report.user_action})")
        save_failure_snapshot(report)

        if report.recovery_action in (RecoveryAction.RETRY_STEP, RecoveryAction.RETRY_PHASE):
            return True
        elif report.recovery_action == RecoveryAction.RETURN_TO_LOBBY:
            if not hwnd:
                hwnd = self._current_hwnd
            if report.code in ("ROBLOX_DISCONNECTED", "ROBLOX_CRASHED"):
                self._attempt_rejoin(hwnd, self._stop_event)
            else:
                self._recover_to_lobby(hwnd, self._stop_event)
            return False
        elif report.recovery_action in (RecoveryAction.STOP_TASK, RecoveryAction.STOP_RUNNER):
            self._set_status(action="Stopped due to failure")
            if report.recovery_action == RecoveryAction.STOP_RUNNER:
                if self._stop_event is not None:
                    self._stop_event.set()
            return False
        return False

    def _run_task_setup(self, hwnd, stop_event: threading.Event, task: dict, mode: str, map_name: str,
                          coords: dict, scroll_power: int, scroll_nudges: int, webhook: dict = None) -> bool:
        """Lobby -> Play -> Story/Raid -> map -> stage/act -> difficulty ->
        confirm -> matchmaking/solo -> teleport-in. Runs once per TASK, not
        once per repeat -- see the repeat loop in _run. Event mode takes its
        own lobby entry (nav_event -> event_gamemode -> Act) with no map or
        difficulty, then rejoins the shared confirm/Solo/Matchmaking tail."""
        if mode in ("summer", "event"):
            reached_summer = False
            for attempt in range(1, MAP_SELECT_RETRY_ATTEMPTS + 1):
                if self._checkpoint(stop_event):
                    return False
                if attempt > 1:
                    self._log(f"[Macro] Retrying Summer entry from the lobby "
                              f"(attempt {attempt}/{MAP_SELECT_RETRY_ATTEMPTS})...")
                if self._reach_summer_selection(hwnd, stop_event, task, scroll_power, scroll_nudges):
                    reached_summer = True
                    break
                if stop_event.is_set():
                    return False
            if not reached_summer:
                self._log(f"[Macro] Couldn't reach the Summer selection after "
                          f"{MAP_SELECT_RETRY_ATTEMPTS} attempts -- stopping.")
                return False
            if self._checkpoint(stop_event):
                return False
        elif mode == "tournament":
            # Tournament goes through Play like Story/Raid (nav_tournament sits
            # on the gamemode menu, picked instead of Story), but has no map
            # carousel and no difficulty -- picking the type card IS the whole
            # selection. It then rejoins the shared confirm + solo Start tail
            # below (its confirm button is nav_entertournament). Same retried-
            # from-the-lobby loop as the map/event paths, since a failed attempt
            # leaves nothing safe to assume about where we ended up.
            reached_tournament = False
            for attempt in range(1, MAP_SELECT_RETRY_ATTEMPTS + 1):
                if self._checkpoint(stop_event):
                    return False
                if attempt > 1:
                    self._log(f"[Macro] Retrying Tournament entry from the lobby "
                               f"(attempt {attempt}/{MAP_SELECT_RETRY_ATTEMPTS})...")
                if self._reach_tournament_selected(hwnd, stop_event, map_name):
                    reached_tournament = True
                    break
                if stop_event.is_set():
                    return False
            if not reached_tournament:
                self._log(f'[Macro] Couldn\'t reach the Tournament screen after '
                           f'{MAP_SELECT_RETRY_ATTEMPTS} attempts -- stopping.')
                return False
            if self._checkpoint(stop_event):
                return False
        elif mode == "tower":
            # Tower goes through Play like Story/Raid (nav_tower sits on the
            # gamemode menu, picked instead of Story), but has no map carousel
            # and no difficulty. Picking Traitless is optional; Normal Tower is
            # selected by doing nothing before the shared Select Stage + solo
            # Start tail below. Same retried-from-the-lobby loop as the
            # Tournament path, since a failed attempt leaves nothing safe to
            # assume about where we ended up.
            reached_tower = False
            for attempt in range(1, MAP_SELECT_RETRY_ATTEMPTS + 1):
                if self._checkpoint(stop_event):
                    return False
                if attempt > 1:
                    self._log(f"[Macro] Retrying Tower entry from the lobby "
                              f"(attempt {attempt}/{MAP_SELECT_RETRY_ATTEMPTS})...")
                if self._reach_tower_selected(hwnd, stop_event, task):
                    reached_tower = True
                    break
                if stop_event.is_set():
                    return False
            if not reached_tower:
                self._log(f'[Macro] Couldn\'t reach the Tower screen after '
                          f'{MAP_SELECT_RETRY_ATTEMPTS} attempts -- stopping.')
                return False
            if self._checkpoint(stop_event):
                return False
        elif mode == "boss_rush":
            reached_boss_rush = False
            for attempt in range(1, MAP_SELECT_RETRY_ATTEMPTS + 1):
                if self._checkpoint(stop_event):
                    return False
                if attempt > 1:
                    self._log(f"[Macro] Retrying Boss Rush entry from the lobby "
                              f"(attempt {attempt}/{MAP_SELECT_RETRY_ATTEMPTS})...")
                if self._reach_boss_rush_selected(hwnd, stop_event):
                    reached_boss_rush = True
                    break
                if stop_event.is_set():
                    return False
            if not reached_boss_rush:
                self._log(f"[Macro] Couldn't reach Boss Rush after "
                          f"{MAP_SELECT_RETRY_ATTEMPTS} attempts -- stopping.")
                return False
            if self._checkpoint(stop_event):
                return False
            self._log('[Macro] Boss Rush is locked to Hard in-game -- no difficulty click needed.')
        else:
            # Lobby -> Play -> Story/Raid -> map search, retried wholesale from
            # the lobby if the map search fails and backing out succeeds (see
            # _spam_back_until_gone) -- a failed search leaves nothing about
            # "already on the gamemode menu" safe to assume anymore, so each
            # attempt re-checks from scratch rather than resuming partway
            # through.
            reached_map = False
            for attempt in range(1, MAP_SELECT_RETRY_ATTEMPTS + 1):
                if self._checkpoint(stop_event):
                    return False
                if attempt > 1:
                    self._log(f"[Macro] Retrying from the lobby (attempt {attempt}/{MAP_SELECT_RETRY_ATTEMPTS})...")
                if self._reach_map_selected(hwnd, stop_event, map_name, mode, scroll_power, scroll_nudges,
                                            task.get("story_event")):
                    reached_map = True
                    break
                if stop_event.is_set():
                    return False
            if not reached_map:
                self._log(f'[Macro] Couldn\'t reach map "{map_name}" after {MAP_SELECT_RETRY_ATTEMPTS} attempts -- stopping.')
                return False
            if self._checkpoint(stop_event):
                return False

            if mode == "expedition":
                # No stage-row picker to click through -- just the difficulty
                # stepper, straight after the map.
                self._interruptible_sleep(DIFFICULTY_CLICK_DELAY, stop_event)
                self._select_expedition_difficulty(hwnd, stop_event, task.get("difficulty") or "1")
            else:
                stage = task.get("stage") or "1"
                if not self._select_stage(hwnd, stop_event, stage, mode, task.get("story_event")):
                    return False
                if self._checkpoint(stop_event):
                    return False

                # Raid's Acts are locked to Hard in-game, same as Story's
                # Infinite/Mastery (see TASK_DATA.raid.fixedDifficulty) -- no
                # difficulty picker exists for it, so no click happens for it.
                if mode == "raid":
                    self._log(f'[Macro] {mode.replace("_", " ").title()} is locked to Hard in-game -- '
                              'no difficulty click needed.')
                elif stage in SPECIAL_STAGES_NO_DIFFICULTY or (
                        mode == "story" and task.get("story_event") not in (None, "", "Normal")):
                    self._log(f'[Macro] "{stage}" is locked to Hard in-game -- no difficulty click needed.')
                else:
                    # _select_stage already settled (DIFFICULTY_CLICK_DELAY)
                    # right after its own double-click, so the panel/toggle is
                    # done animating in by the time we get here.
                    self._select_difficulty(hwnd, task.get("difficulty") or "Normal", coords)
            if self._checkpoint(stop_event):
                return False

        return self._enter_selected_stage(hwnd, stop_event, task, mode, coords, webhook)

    def _enter_selected_stage(self, hwnd, stop_event: threading.Event, task: dict, mode: str,
                                coords: dict, webhook: dict = None) -> bool:
        """Enter a stage whose final map/stage panel is already open."""
        # nav_select_stage is a confirm button that finalizes the stage/
        # difficulty pick -- Start/Enter Matchmaking doesn't actually
        # appear/work until it's pressed, so it needs an actual (verified,
        # retried) click, not just a wait. Solo-only: matchmaking goes
        # straight to Enter Matchmaking instead, since this doesn't
        # reliably show up the same way for it.
        summer_portal = mode == "summer" and task.get("summer_mode") == "Portal Mode"
        story_event = mode == "story" and task.get("story_event") not in (None, "", "Normal")
        if task.get("play_mode") != "matchmaking" and not summer_portal:
            if mode == "tournament":
                confirm_image = "nav_entertournament"
            elif mode == "expedition":
                confirm_image = "exp_select_stage"
            else:
                confirm_image = "nav_select_stage"
            self._set_status(action="Clicking Select Stage...")
            if not self._click_and_verify_gone(hwnd, stop_event, confirm_image, STAGE_SCREEN_TIMEOUT):
                self._log(f'[Macro] "{confirm_image}" never showed up -- stopping.')
                return False
            if mode == "boss_rush":
                self._log(f"[Boss Rush] Waiting {BOSS_RUSH_TRANSITION_DELAY:.0f}s for the stage confirmation to settle.")
                self._interruptible_sleep(BOSS_RUSH_TRANSITION_DELAY, stop_event)
        if self._checkpoint(stop_event):
            return False

        # Solo has its own direct Start button -- Enter Matchmaking is
        # multiplayer-only and was never going to appear in Solo mode, which
        # is exactly why it kept sitting there waiting on it and looking
        # like it was "going to matchmaking" regardless of this setting.
        if task.get("play_mode") == "matchmaking":
            if not self._click_enter_matchmaking(hwnd, stop_event, coords, mode, story_event=story_event):
                return False
            if self._checkpoint(stop_event):
                return False
            self._log(f"[Macro] Waiting for the lobby to fill (up to {MATCHMAKING_TELEPORT_TIMEOUT / 60:.0f} "
                       f"min) -- matchmaking has to find real players before it teleports in.")
            if not self._wait_teleport_in(hwnd, stop_event, webhook, task, timeout=MATCHMAKING_TELEPORT_TIMEOUT):
                return False
        else:
            if mode == "boss_rush":
                self._log(f"[Boss Rush] Waiting {BOSS_RUSH_TRANSITION_DELAY:.0f}s before starting the Solo run.")
                self._interruptible_sleep(BOSS_RUSH_TRANSITION_DELAY, stop_event)
                if self._checkpoint(stop_event):
                    return False
            self._log("[Macro] Solo mode -- clicking Start (retrying up to "
                       f"{SOLO_START_RETRY_ATTEMPTS} times if it doesn't teleport).")
            if not self._click_start_and_wait_teleport(hwnd, stop_event, webhook, task):
                return False
        return not self._checkpoint(stop_event)

    def _play_one_match(self, hwnd, stop_event: threading.Event, task: dict, default_walk_paths: dict,
                          first_repeat: bool = True, webhook: dict = None):
        """Assumes teleport-in already happened -- the initial one from
        _run_task_setup, or a repeat's re-teleport after Repeat Stage (see
        _handle_match_result). Start Game settings check, Pre Start, the
        actual Start Game click, then watches for Victory/Defeat. Runs once
        per repeat. first_repeat gates the default walk and any "Once"
        Pre Start block so they only fire on the task's first entry into
        this stage, not on every repeat (see _run_prestart). Returns
        "win"/"loss", "configuration_error" for a missing Boss Rush setup
        reference/route, or None on other failure/stop."""
        self._active_task = task
        boss_rush_state = None
        if task.get("mode") == "boss_rush":
            boss_rush_state = self._new_boss_rush_state(task)
            if boss_rush_state is None:
                self._log("[Boss Rush] Invalid gate count or card preference. Set gates to 1-6 and choose left, middle, or right.")
                return None
        if not self._start_game_or_reset_via_settings(hwnd, stop_event, task.get("play_mode")):
            return None
        if self._checkpoint(stop_event):
            return None

        defer_boss_rush_blocks = boss_rush_state is not None
        if not self._run_prestart(
            hwnd, stop_event, task, default_walk_paths, first_repeat,
            run_blocks=not defer_boss_rush_blocks,
        ):
            return None
        if self._checkpoint(stop_event):
            return None

        self._log("[Macro] Pre Start setup finished -- starting the round.")
        self._set_status(action="Starting the round...")
        if self._checkpoint(stop_event):
            return None
        # Start Game genuinely applies to Expedition too (it can show up
        # more than once, similar to Infinite mode) -- not skipped here.
        self._wait_out_start_game_warning(hwnd, stop_event)
        if self._checkpoint(stop_event):
            return None
        start_name, start_match = self._find_start_game_button(hwnd, stop_event, START_GAME_BUTTON_WAIT_TIMEOUT)
        if start_match is None:
            # Not fatal: Start Game may already have been pressed by the
            # leader (or Auto Vote Start already handles it) earlier in
            # _start_game_or_reset_via_settings -- its absence here just
            # means the round is already starting on its own.
            self._log("[Macro] Start Game not found -- already started, continuing.")
        else:
            for attempt in range(1, START_GAME_CLICK_RETRY_ATTEMPTS + 1):
                debug_path = self._debug_save(hwnd, start_name, start_match)
                suffix = f" Debug: {debug_path}" if debug_path else ""
                self._log(f"[Macro] Found Start Game ({start_name}, score {start_match['score']:.2f}) -- "
                           f"clicking it (attempt {attempt}/{START_GAME_CLICK_RETRY_ATTEMPTS}).{suffix}")
                if not wm.activate_window(hwnd):
                    self._log("[Macro] Couldn't confirm focus before clicking Start Game -- "
                               "click may not register.")
                # Z first -- same deselect pressed before every unit placement:
                # if Pre Start's last block left a unit selected/placing, the
                # cursor still owns that state and the Start Game click gets
                # eaten by it instead of landing on the button. (After
                # activate_window, so the tap actually reaches the game.)
                self._keyboard.tap(ord("Z"))
                time.sleep(0.1)
                vision.click_match(self._mouse, hwnd, start_match)
                self._interruptible_sleep(START_GAME_CLICK_VERIFY_SETTLE, stop_event)
                if self._checkpoint(stop_event):
                    return None

                start_name, start_match = self._find_start_game_button(hwnd)
                if start_match is None:
                    break
                if attempt == START_GAME_CLICK_RETRY_ATTEMPTS:
                    self._log(f"[Macro] Start Game still showing after {START_GAME_CLICK_RETRY_ATTEMPTS} "
                               f"clicks -- the round may not have actually started. Continuing anyway.")
                    screenshot_path = self._save_debug_screenshot_unconditional(hwnd, "start_game_click_stuck")
                    self._send_event_webhook(
                        webhook, task, "Start Game Click Not Registering",
                        f"Clicked Start Game {START_GAME_CLICK_RETRY_ATTEMPTS} times but it's still showing -- "
                        f"the round may not have actually started.", 0xE05A6D, screenshot_path)
        if self._checkpoint(stop_event):
            return None

        # Team Loadout (including its saved-equipment checkbox setting) is
        # applied earlier in Pre Start -- see _apply_team_loadout.
        self._set_status(action="Battle...")
        self._log("[Macro] Moving into Battle.")

        battle_blocks = self._load_battle_blocks(task)
        self._battle_block_index = 0
        self._battle_block_state = {}
        # Battle clock for the "Leave at Minute" block (see runner_blocks).
        # Reset per match so the minute is measured from THIS battle's start.
        self._battle_started_at = time.time()
        self._battle_leave_requested = False
        self._is_expedition_match = task.get("mode") == "expedition"
        self._wave_region = EXPEDITION_WAVE_REGION if self._is_expedition_match else WAVE_REGION
        self._last_reward_card_at = 0.0
        self._last_board_disruption_at = 0.0
        self._battle_replayed = False
        # Loop A / Loop B: their own index+state, ticked and restarted every
        # poll alongside Battle (see _tick_loop_phases).
        loop_blocks = self._load_loop_blocks(task)
        self._loop_runtime = {key: {"blocks": loop_blocks[key], "index": 0, "state": {},
                                    "completed_detects": set()}
                              for key in ("loop_a", "loop_b")}
        self._release_quick_place_shift()  # safety net -- never enter a match with Shift stuck down from before
        # exp_extract is a recurring checkpoint choice (Extract AND Continue
        # offered side by side), not a one-shot terminal event -- confirmed
        # from a real test run where extract_after=2 still extracted on the
        # very first sighting instead of continuing through 2 checkpoints
        # first. So this DOES need real counting: decline (click the
        # "exp_extract_continue" choice on this same screen) every sighting
        # up to extract_after, only accept the one right after that.
        self._expedition_extract_count = 0
        # Checkpoints where extraction was attempted and did not take. Past
        # EXPEDITION_EXTRACT_ATTEMPTS_BEFORE_PLAYING_ON the run stops asking
        # and plays itself out -- the host may simply not be extracting.
        self._exp_failed_extracts = 0
        self._expedition_extract_accept_at = _parse_extract_after(
            task.get("extract_after")) + 1
        self._exp_last_sighting_at = 0.0  # fresh match, fresh sighting-debounce clock (see EXP_COLOR_SIGHTING_DEBOUNCE)
        # ...and fresh no-progress clocks (see EXPEDITION_STALL_TIMEOUT and
        # EXPEDITION_INTERCEPT_TIMEOUT).
        self._exp_checkpoint_streak = 0
        self._exp_checkpoint_since = 0.0
        self._exp_intercept_streak = 0
        self._exp_intercept_since = 0.0
        self._exp_clock_marked_at = 0.0
        # Spirit City Act 3's boss/cutscene "Click anywhere to close" popup
        # (see _click_close_popup_if_found) only ever shows up there.
        watch_close_popup = (task.get("mode") == "raid" and task.get("map") == "Spirit City"
                              and str(task.get("stage")) == "3")
        if boss_rush_state is not None:
            self._interruptible_sleep(2.0, stop_event)
            if self._checkpoint(stop_event):
                return None
            self._log("[Boss Rush] Starting the configured Spawn → Gate 1 route.")
            if not self._run_boss_rush_gate_route(hwnd, stop_event, task, 1, boss_rush_state):
                return "configuration_error" if boss_rush_state["configuration_error"] else None
            if self._checkpoint(stop_event):
                return None
            if not self._run_boss_rush_gate_setup(hwnd, stop_event, task, 1, boss_rush_state):
                return "configuration_error" if boss_rush_state["configuration_error"] else None
        return self._wait_for_match_result(hwnd, stop_event, battle_blocks, first_repeat, task.get("macro"),
                                             task.get("mode"), watch_close_popup, webhook, task, boss_rush_state)

    @staticmethod
    def _new_boss_rush_state(task: dict):
        value = task.get("boss_rush_gates", 6)
        if isinstance(value, bool):
            return None
        try:
            target = int(value)
        except (TypeError, ValueError):
            return None
        if target < 1 or target > 6:
            return None
        card = task.get("boss_rush_card", "left")
        if card not in BOSS_RUSH_CARD_COORDS:
            return None
        return {
            "target": target,
            "card": card,
            "cleared": 0,
            "awaiting_decision": False,
            "decision_since": 0.0,
            "card_click_attempts": 0,
            "card_last_click_at": 0.0,
            "boss_started": False,
            "configuration_error": False,
            "missing_template_logged": set(),
        }

    def _run_boss_rush_gate_route(self, hwnd, stop_event: threading.Event, task: dict, gate: int,
                                  state: dict = None) -> bool:
        route_name = (task.get("boss_rush_walk_paths") or {}).get(str(gate)) or ""
        if not route_name:
            self._log(f"[Boss Rush] Gate {gate} has no Spawn → Gate route. Assign it in Resource > Boss Rush Routes.")
            if state is not None:
                state["configuration_error"] = True
            return False
        events = paths.load_path(route_name).get("events") or []
        if not events:
            self._log(f'[Boss Rush] Gate {gate} route "{route_name}" is missing or empty. Record and assign it in Resource > Boss Rush Routes.')
            if state is not None:
                state["configuration_error"] = True
            return False
        if not self._keyboard or not self._mouse:
            self._log("[Boss Rush] Keyboard and mouse input are required to replay a gate route and interact with the gate.")
            if state is not None:
                state["configuration_error"] = True
            return False
        if self._checkpoint(stop_event):
            return False
        self._log(f'[Boss Rush] Replaying Spawn → Gate {gate} route "{route_name}" ({len(events)} events).')
        paths.replay_events(events, self._keyboard, stop_event)
        if self._checkpoint(stop_event):
            return False
        if not wm.activate_window(hwnd):
            self._log("[Boss Rush] Couldn't confirm game focus before interacting with the gate.")
        self._keyboard.tap(ord("E"))
        self._interruptible_sleep(0.5, stop_event)
        return not self._checkpoint(stop_event)

    def _run_boss_rush_gate_setup(self, hwnd, stop_event: threading.Event, task: dict, gate: int,
                                  state: dict) -> bool:
        if gate == 1:
            self._log(f"[Boss Rush] Gate {gate} route complete -- running Pre Start placement blocks.")
            self._run_prestart_blocks(
                hwnd, stop_event, task, first_repeat=True,
                default_walk_paths={}, skip_walk_paths=True,
            )
        else:
            self._log(f"[Boss Rush] Gate {gate} route complete -- skipping Pre Start blocks.")
        if self._checkpoint(stop_event):
            return False
        return self._click_boss_rush_gate_start(hwnd, stop_event, gate, state)

    def _click_boss_rush_gate_start(self, hwnd, stop_event: threading.Event, gate, state: dict) -> bool:
        label = f"Gate {gate}" if isinstance(gate, int) else str(gate)
        self._wait_out_start_game_warning(hwnd, stop_event)
        if self._checkpoint(stop_event):
            return False
        start_name, start_match = self._find_start_game_button(
            hwnd, stop_event, START_GAME_BUTTON_WAIT_TIMEOUT)
        if start_match is None:
            if not stop_event.is_set():
                self._log(f'[Boss Rush] Start Game was not found after {label} placement; cannot start this gate.')
                state["configuration_error"] = True
            return False

        for attempt in range(1, START_GAME_CLICK_RETRY_ATTEMPTS + 1):
            self._log(
                f'[Boss Rush] Found Start Game for {label} ({start_name}, '
                f'score {start_match["score"]:.2f}) -- clicking it (attempt '
                f'{attempt}/{START_GAME_CLICK_RETRY_ATTEMPTS}).'
            )
            if not wm.activate_window(hwnd):
                self._log(f"[Boss Rush] Couldn't confirm game focus before starting {label}.")
            self._keyboard.tap(ord("Z"))
            self._interruptible_sleep(0.1, stop_event)
            if self._checkpoint(stop_event):
                return False
            vision.click_match(self._mouse, hwnd, start_match)
            self._interruptible_sleep(START_GAME_CLICK_VERIFY_SETTLE, stop_event)
            if self._checkpoint(stop_event):
                return False
            start_name, start_match = self._find_start_game_button(hwnd)
            if start_match is None:
                self._log(f"[Boss Rush] {label} started.")
                return True

        self._log(f"[Boss Rush] Start Game stayed visible after {START_GAME_CLICK_RETRY_ATTEMPTS} "
                  f"clicks for {label}; stopping to avoid proceeding before it starts.")
        state["configuration_error"] = True
        return False

    def _start_boss_rush_boss_setup(self, hwnd, stop_event: threading.Event, task: dict,
                                    state: dict = None) -> bool:
        macro_name = task.get("boss_rush_boss_macro")
        if not macro_name:
            self._log("[Boss Rush] Boss Setup Macro is not selected. Choose a placement-only macro in the Task settings before running Boss Rush.")
            if state is not None:
                state["configuration_error"] = True
            return False
        self._interruptible_sleep(2.5, stop_event)
        if self._checkpoint(stop_event):
            return False
        boss_task = {**task, "macro": macro_name}
        self._log(f'[Boss Rush] Running Boss Setup Macro "{macro_name}" to place units for the boss.')
        self._active_task = boss_task
        self._run_prestart_blocks(
            hwnd, stop_event, boss_task, first_repeat=True,
            default_walk_paths={}, skip_walk_paths=True,
        )
        self._active_task = task
        if self._checkpoint(stop_event):
            return False
        if state is None:
            self._log("[Boss Rush] Cannot start the boss gate without gate state.")
            return False
        return self._click_boss_rush_gate_start(hwnd, stop_event, "Boss Gate", state)

    def _click_boss_rush_card(self, hwnd, stop_event: threading.Event, state: dict) -> bool:
        if state["card_click_attempts"] >= BOSS_RUSH_CARD_CLICK_ATTEMPTS:
            self._log(
                f'[Boss Rush] Could not confirm the "{state["card"]}" card after '
                f"{BOSS_RUSH_CARD_CLICK_ATTEMPTS} clicks; returning to the lobby."
            )
            screenshot = self._save_debug_screenshot_unconditional(
                hwnd, "boss_rush_card_click_failed")
            if screenshot:
                self._log(f"[Boss Rush] Card-selection screenshot: {screenshot}")
            return False
        x_key, y_key = BOSS_RUSH_CARD_COORDS[state["card"]]
        try:
            ref_x, ref_y = float(self._coords[x_key]), float(self._coords[y_key])
        except (KeyError, TypeError, ValueError):
            self._log(f'[Boss Rush] The saved {state["card"]} card coordinate is invalid. '
                      "Set it with Pick in Settings > Debug > Macro Coordinates.")
            state["configuration_error"] = True
            return False
        if not (0 <= ref_x < config.FIXED_WIN_W and 0 <= ref_y < config.FIXED_WIN_H):
            self._log(f'[Boss Rush] The saved {state["card"]} card coordinate '
                      f"({ref_x:g}, {ref_y:g}) is outside the game window. Re-pick it in "
                      "Settings > Debug > Macro Coordinates.")
            state["configuration_error"] = True
            return False

        card_x, card_y = vision.ref_to_screen(hwnd, ref_x, ref_y)
        if self._checkpoint(stop_event):
            return False
        if not wm.activate_window(hwnd):
            self._log("[Boss Rush] Couldn't confirm game focus before selecting the card.")
        attempt = state["card_click_attempts"] + 1
        self._log(
            f'[Boss Rush] Clicking the configured "{state["card"]}" card at '
            f"({ref_x:g}, {ref_y:g}) (attempt {attempt}/{BOSS_RUSH_CARD_CLICK_ATTEMPTS})."
        )
        self._mouse.click(card_x, card_y)
        state["card_click_attempts"] = attempt
        state["card_last_click_at"] = time.monotonic()

        deadline = state["card_last_click_at"] + BOSS_RUSH_CARD_CLICK_VERIFY_TIMEOUT
        while time.monotonic() < deadline:
            if self._checkpoint(stop_event):
                return False
            try:
                prompt = vision.find_image(hwnd, "boss_rush_pick_card")
            except vision.TemplateNotFound as exc:
                self._log(f"[Boss Rush] {exc}")
                state["configuration_error"] = True
                return False
            if prompt is None:
                return True
            self._interruptible_sleep(
                min(0.25, deadline - time.monotonic()), stop_event)
        self._log(
            f"[Boss Rush] Pick Card is still visible after click "
            f"{attempt}/{BOSS_RUSH_CARD_CLICK_ATTEMPTS}; will verify again before retrying."
        )
        return True

    def _wait_for_boss_rush_decision_gone(self, hwnd, image_name: str,
                                          stop_event: threading.Event, state: dict) -> bool:
        deadline = time.monotonic() + BOSS_RUSH_DECISION_CLOSE_TIMEOUT
        while time.monotonic() < deadline:
            if self._checkpoint(stop_event):
                return False
            try:
                match = vision.find_image(hwnd, image_name)
                if match is None:
                    match = vision.find_image(
                        hwnd, image_name, threshold=BOSS_RUSH_DECISION_FALLBACK_THRESHOLD)
            except vision.TemplateNotFound as exc:
                self._log(f"[Boss Rush] {exc}")
                state["configuration_error"] = True
                return False
            if match is None:
                return True
            self._interruptible_sleep(0.3, stop_event)

        self._log(
            f'[Boss Rush] "{image_name}" remained visible after clicking it; '
            "not starting another gate until the choice is confirmed."
        )
        screenshot = self._save_debug_screenshot_unconditional(
            hwnd, "boss_rush_decision_click_not_confirmed")
        if screenshot:
            self._log(f"[Boss Rush] Decision-click screenshot: {screenshot}")
        return False

    def _retry_boss_rush_card_if_needed(self, hwnd, stop_event: threading.Event,
                                        state: dict):
        if time.monotonic() - state["card_last_click_at"] < BOSS_RUSH_CARD_CLICK_RETRY_DELAY:
            return "handled"
        if state["card_click_attempts"] >= BOSS_RUSH_CARD_CLICK_ATTEMPTS:
            self._log(
                f'[Boss Rush] Could not confirm the "{state["card"]}" card after '
                f"{BOSS_RUSH_CARD_CLICK_ATTEMPTS} clicks; returning to the lobby."
            )
            screenshot = self._save_debug_screenshot_unconditional(
                hwnd, "boss_rush_card_click_failed")
            if screenshot:
                self._log(f"[Boss Rush] Card-selection screenshot: {screenshot}")
            return "failed"
        self._log("[Boss Rush] Pick Card is still visible; retrying the configured card click.")
        if not self._click_boss_rush_card(hwnd, stop_event, state):
            return "failed"
        return "handled"

    def _handle_boss_rush_progress(self, hwnd, stop_event: threading.Event, task: dict, state: dict):
        if state["boss_started"]:
            return "idle"

        try:
            card_prompt = vision.find_image(hwnd, "boss_rush_pick_card")
        except vision.TemplateNotFound as exc:
            if "boss_rush_pick_card" not in state["missing_template_logged"]:
                self._log(f"[Boss Rush] {exc} Add an Image Manager reference named boss_rush_pick_card for the Pick Card prompt.")
                state["missing_template_logged"].add("boss_rush_pick_card")
            state["configuration_error"] = True
            return "failed"
        if card_prompt is not None and not state["awaiting_decision"]:
            state["card_click_attempts"] = 0
            state["card_last_click_at"] = 0.0
            self._log(f'[Boss Rush] Pick Card found; selecting the configured "{state["card"]}" card.')
            if not self._click_boss_rush_card(hwnd, stop_event, state):
                return "failed"
            state["cleared"] += 1
            state["awaiting_decision"] = True
            state["decision_since"] = time.time()
            return "handled"

        if not state["awaiting_decision"]:
            return "idle"

        cleared = state["cleared"]
        target = state["target"]
        if cleared == 1 and cleared < target:
            if card_prompt is not None:
                return self._retry_boss_rush_card_if_needed(hwnd, stop_event, state)
            next_gate = cleared + 1
            self._log(
                f"[Boss Rush] Gate 1 card selected; the game automatically returns to "
                f"the gates map. Preparing Gate {next_gate} without waiting for Continue."
            )
            self._interruptible_sleep(1.5, stop_event)
            if self._checkpoint(stop_event):
                return "failed"
            if not self._run_boss_rush_gate_route(hwnd, stop_event, task, next_gate, state):
                return "failed"
            if not self._run_boss_rush_gate_setup(hwnd, stop_event, task, next_gate, state):
                return "failed"
            state["awaiting_decision"] = False
            return "handled"

        action = "continue" if cleared < target or (cleared == 6 and target == 6) else "fight_boss"
        image_name = f"boss_rush_{action}"
        try:
            action_match = vision.find_image(hwnd, image_name)
        except vision.TemplateNotFound as exc:
            if image_name not in state["missing_template_logged"]:
                self._log(f"[Boss Rush] {exc} Add an Image Manager reference named {image_name} for this decision.")
                state["missing_template_logged"].add(image_name)
            state["configuration_error"] = True
            return "failed"
        elapsed = time.time() - state["decision_since"]
        if action_match is None and elapsed >= BOSS_RUSH_DECISION_FALLBACK_DELAY:
            try:
                action_match = vision.find_image(
                    hwnd, image_name, threshold=BOSS_RUSH_DECISION_FALLBACK_THRESHOLD)
            except vision.TemplateNotFound as exc:
                if image_name not in state["missing_template_logged"]:
                    self._log(f"[Boss Rush] {exc} Add an Image Manager reference named {image_name} for this decision.")
                    state["missing_template_logged"].add(image_name)
                state["configuration_error"] = True
                return "failed"
            if action_match is not None:
                self._log(
                    f'[Boss Rush] Matched "{image_name}" with the transition fallback '
                    f"(score {action_match['score']:.2f})."
                )
        if action_match is None and card_prompt is not None:
            return self._retry_boss_rush_card_if_needed(hwnd, stop_event, state)
        if action_match is not None:
            max_click_attempts = (
                BOSS_RUSH_CONTINUE_CLICK_ATTEMPTS if action == "continue" else 1
            )
            for attempt in range(1, max_click_attempts + 1):
                if attempt == 1:
                    self._log(f'[Boss Rush] Selecting "{action.replace("_", " ").title()}" after gate {cleared}.')
                else:
                    self._log(
                        f'[Boss Rush] Retrying "{action.replace("_", " ").title()}" click '
                        f"({attempt}/{max_click_attempts})."
                    )
                if not wm.activate_window(hwnd):
                    self._log("[Boss Rush] Couldn't confirm game focus before selecting the gate decision.")
                vision.click_match(self._mouse, hwnd, action_match)
                if self._wait_for_boss_rush_decision_gone(hwnd, image_name, stop_event, state):
                    break
                if stop_event.is_set():
                    return "failed"
                if attempt == max_click_attempts:
                    return "failed"

                self._interruptible_sleep(0.5, stop_event)
                if self._checkpoint(stop_event):
                    return "failed"
                try:
                    action_match = vision.find_image(hwnd, image_name)
                    if action_match is None:
                        action_match = vision.find_image(
                            hwnd, image_name,
                            threshold=BOSS_RUSH_DECISION_FALLBACK_THRESHOLD)
                except vision.TemplateNotFound as exc:
                    self._log(f"[Boss Rush] {exc}")
                    state["configuration_error"] = True
                    return "failed"
                if action_match is None:
                    self._log(
                        f'[Boss Rush] "{image_name}" disappeared after the click; '
                        "continuing with the gate transition."
                    )
                    break
            state["awaiting_decision"] = False
            if action == "continue":
                if cleared == 6:
                    if not self._start_boss_rush_boss_setup(hwnd, stop_event, task, state):
                        return "failed"
                    state["boss_started"] = True
                    return "handled"
                next_gate = cleared + 1
                self._interruptible_sleep(1.5, stop_event)
                if self._checkpoint(stop_event):
                    return "failed"
                if not self._run_boss_rush_gate_route(hwnd, stop_event, task, next_gate, state):
                    return "failed"
                if not self._run_boss_rush_gate_setup(hwnd, stop_event, task, next_gate, state):
                    return "failed"
            else:
                if not self._start_boss_rush_boss_setup(hwnd, stop_event, task, state):
                    return "failed"
                state["boss_started"] = True
            return "handled"

        if elapsed >= BOSS_RUSH_DECISION_TIMEOUT:
            self._log(
                f'[Boss Rush] The "{image_name}" prompt was not detected after '
                f"{BOSS_RUSH_DECISION_TIMEOUT:.0f}s. Check the Image Manager crop and its "
                "match sensitivity; stopping this Boss Rush task without returning to the lobby."
            )
            screenshot = self._save_debug_screenshot_unconditional(
                hwnd, "boss_rush_decision_not_found")
            if screenshot:
                self._log(f"[Boss Rush] Decision-screen screenshot: {screenshot}")
            state["configuration_error"] = True
            return "failed"
        return "handled"


    @staticmethod
    def _infinite_wave_limit(task: dict):
        """Configured completed-wave target for a Story > Infinite task."""
        task = task or {}
        is_event_wave = (
            task.get("mode") == "summer"
            and task.get("summer_mode") == "Event Mode"
        )
        if not (
            (task.get("mode") == "story" and task.get("stage") == "Infinite")
            or is_event_wave
        ):
            return None
        try:
            return max(1, int(task.get("infinite_wave_limit") or DEFAULT_INFINITE_WAVE_LIMIT))
        except (TypeError, ValueError):
            return DEFAULT_INFINITE_WAVE_LIMIT

    def _leave_infinite_at_wave_limit(self, hwnd, stop_event: threading.Event, limit: int) -> bool:
        """Leave a live Infinite match after ``limit`` has fully completed."""
        self._release_quick_place_shift()
        self._set_status(action=f"Wave {limit} complete -- leaving stage...")
        self._log(
            f"[Macro] Infinite wave {limit} completed and wave {limit + 1} began -- "
            "clicking Leave Stage."
        )
        if not self._click_and_verify_gone(
                hwnd, stop_event, "leave_stage", NAV_CLICK_TIMEOUT, success_name="return"):
            self._log('[Macro] "Leave Stage" not found after reaching the Infinite wave limit.')
            return False
        self._click_return_to_lobby_if_found(hwnd, stop_event)
        return not self._checkpoint(stop_event)

    def _check_infinite_wave_limit(self, hwnd, stop_event: threading.Event, limit: int, state: dict):
        """Poll/confirm the unlimited-wave HUD and leave at ``limit + 1``.

        Returns ``"wave_limit"`` after a successful exit, ``"failed"`` if
        the exit UI could not be used, and ``None`` while the match continues.
        """
        now = time.time()
        if now < state.get("next_check", 0.0):
            return None
        state["next_check"] = now + WAIT_WAVE_POLL_INTERVAL

        try:
            image = vision.capture_game_bgr(hwnd, self._wave_region)
            if image is None:
                raise RuntimeError("window capture returned no image")
            current, maximum = wave_module.read_wave(image)
        except Exception as exc:
            state.pop("confirmations", None)
            if not state.get("read_error_logged"):
                self._log(
                    f"[Macro] Couldn't read the Infinite wave counter ({exc}) -- "
                    f"retrying every {WAIT_WAVE_POLL_INTERVAL:.0f}s."
                )
                state["read_error_logged"] = True
            return None

        if current is None or maximum not in (None, "∞"):
            state.pop("confirmations", None)
            return None
        state.pop("read_error_logged", None)
        self._set_status(action=f"Infinite wave {current} -- leaving after wave {limit}...")

        exit_wave = limit + 1
        if current < exit_wave:
            # Remember that the requested final wave was genuinely observed.
            # This is the evidence that lets a later confirmed reading stand
            # in for an exit-wave frame OCR happened to miss.
            if current == limit:
                state["target_wave_seen"] = True
            state.pop("confirmations", None)
            state.pop("confirmation_wave", None)
            return None
        if current > exit_wave and not state.get("target_wave_seen"):
            # Do not let an isolated leading-digit hallucination (for
            # example wave 5 read as 55) end a run early. Exact exit-wave
            # reads remain valid on their own; an overshoot is accepted only
            # after the real target wave was seen earlier in this match.
            state.pop("confirmations", None)
            state.pop("confirmation_wave", None)
            return None

        confirmations = (
            state.get("confirmations", 0) + 1
            if state.get("confirmation_wave") == current
            else 1
        )
        state["confirmation_wave"] = current
        state["confirmations"] = confirmations
        if confirmations < INFINITE_WAVE_LIMIT_CONFIRMATIONS:
            self._log(
                f"[Macro] Infinite wave {current} reached the exit point -- "
                "confirming on the next read."
            )
            return None
        return (
            "wave_limit"
            if self._leave_infinite_at_wave_limit(hwnd, stop_event, limit)
            else "failed"
        )



    def _wait_for_match_result(self, hwnd, stop_event: threading.Event, battle_blocks: list = None,
                                 first_repeat: bool = True, macro_name: str = None, mode: str = None,
                                 watch_close_popup: bool = False, webhook: dict = None, task: dict = None,
                                 boss_rush_state: dict = None) -> str:
        boss_rush_watch_logged = False
        if boss_rush_state is None:
            self._log("[Macro] Battle in progress -- watching for Victory/Defeat...")
            self._set_status(action="Battle in progress...")
        else:
            self._set_status(action="Boss Rush gates in progress...")
        battle_blocks = battle_blocks or []
        infinite_wave_limit = self._infinite_wave_limit(task)
        infinite_wave_state = {}
        if infinite_wave_limit is not None:
            # A configured Infinite run can legitimately take longer than
            # the finite-stage 30-minute safety timeout.
            deadline = None
            self._log(
                f"[Macro] Infinite wave limit: finish wave {infinite_wave_limit}, "
                f"then leave when wave {infinite_wave_limit + 1} begins."
            )
        else:
            deadline = time.time() + MATCH_RESULT_TIMEOUT
        lobby_sightings = 0   # consecutive polls that found the lobby's Play button
        afk_clicked_at = 0.0  # last time the AFK Chamber exit was clicked
        # {"handled_at", "seen_at"} -- the settle is deferred, not slept, so the
        # poll loop keeps picking upgrade cards and clicking Continues meanwhile.
        encounter_state = {"handled_at": 0.0, "seen_at": 0.0}
        summer_portal_card_state = {"selected": False}
        while deadline is None or time.time() < deadline:
            if self._checkpoint(stop_event):
                return None

            if boss_rush_state is not None:
                boss_result = self._handle_boss_rush_progress(hwnd, stop_event, task, boss_rush_state)
                if boss_result == "failed":
                    return "configuration_error" if boss_rush_state["configuration_error"] else None
                if boss_rush_state["boss_started"] and not boss_rush_watch_logged:
                    self._log("[Macro] Boss battle in progress -- watching for Victory/Defeat...")
                    self._set_status(action="Boss battle in progress...")
                    boss_rush_watch_logged = True
                if boss_result == "handled":
                    self._interruptible_sleep(MATCH_RESULT_POLL_INTERVAL, stop_event)
                    continue

            # Eclipse Infinite presents the Pick Card modal repeatedly during
            # the battle (every five waves). Check it on every live poll so
            # the modal is handled even when no Battle block or result screen
            # happens to trigger a card check.
            if (task and task.get("mode") == "story"
                    and task.get("story_event") == "Eclipsed Infinite"):
                if self._dismiss_reward_card_if_found(hwnd):
                    if self._checkpoint(stop_event):
                        return None
                    self._interruptible_sleep(MATCH_RESULT_POLL_INTERVAL, stop_event)
                    continue

            # Leaving an Infinite run at its requested wave is a hard task
            # boundary, so check it before Battle blocks. Upgrade Unit can
            # spend several seconds waiting for its info panel; putting the
            # wave check after that work made the narrow exit-wave window
            # much easier to miss.
            if infinite_wave_limit is not None:
                limit_result = self._check_infinite_wave_limit(
                    hwnd, stop_event, infinite_wave_limit, infinite_wave_state)
                if limit_result == "wave_limit":
                    return "wave_limit"
                if limit_result == "failed":
                    return None

            if battle_blocks:
                self._run_battle_blocks_tick(hwnd, stop_event, battle_blocks, first_repeat, macro_name)
                if self._checkpoint(stop_event):
                    return None
            # Loop phases cycle continuously for the whole match, not once
            # through -- ticked here every poll right after Battle.
            self._tick_loop_phases(hwnd, stop_event, first_repeat, macro_name)
            if self._checkpoint(stop_event):
                return None
            if (task and task.get("mode") == "summer"
                    and task.get("summer_mode") == "Portal Mode"
                    and not summer_portal_card_state["selected"]):
                if self._select_summer_portal_card(hwnd, stop_event, task):
                    summer_portal_card_state["selected"] = True
                    if self._checkpoint(stop_event):
                        return None
            # A "Leave at Minute" block (Battle or a Loop phase) hit its time
            # and already left the live match to the lobby (clicked To Lobby ->
            # Return). Checked here, after BOTH the Battle and Loop ticks, so it
            # works from either -- and even with an empty Battle phase. Report
            # "left" so _run_task skips the Victory/Defeat handling and re-
            # enters from the lobby (same path as the Infinite wave-limit exit,
            # see left_live_match).
            if self._battle_leave_requested:
                self._battle_leave_requested = False
                return "left"

            # Roblox's own Reconnect/Retry prompt can show up mid-battle too,
            # not just during the teleport-in wait -- this used to only be
            # checked there, so a disconnect that happened AFTER teleporting
            # in successfully was invisible to the macro entirely: it just
            # kept polling for Victory/Defeat/exp_continue/exp_extract
            # against a dead screen until MATCH_RESULT_TIMEOUT gave up,
            # instead of recognizing the disconnect and rejoining promptly.
            for name in RECONNECT_IMAGE_NAMES:
                try:
                    reconnect_match = vision.find_image(hwnd, name)
                except vision.TemplateNotFound:
                    continue
                if reconnect_match is not None:
                    self._handle_disconnect(hwnd, stop_event, webhook, task)
                    return None

            # Portal Victory renders over the stage/lobby background, so the
            # lobby Play image can remain visible behind the result panel.
            # Check the authoritative Select Portal button before using Play
            # as a lobby-resync signal.
            if (task and task.get("mode") == "summer"
                    and task.get("summer_mode") == "Portal Mode"):
                try:
                    select_portal_match = vision.find_image(hwnd, SUMMER_SELECT_PORTAL_IMAGE)
                except vision.TemplateNotFound as exc:
                    self._log(f"[Macro] {exc}")
                    return None
                if select_portal_match is not None:
                    self._log(
                        f'[Macro] Portal Victory screen detected via "Select Portal" '
                        f'(score {select_portal_match["score"]:.2f}).'
                    )
                    return "win"

            # Back on the lobby mid-match. Nothing the runner did put us here:
            # someone clicked Return to Lobby by hand, or the game ejected us.
            # Either way Victory/Defeat can never arrive, so without this the
            # run just polls a lobby screen until MATCH_RESULT_TIMEOUT.
            # nav_play only renders on the lobby, which is what makes it a safe
            # signal from inside a match. Reported as "left" -- the same result
            # Leave at Minute and the Infinite wave-limit exit use, so _run_task
            # re-runs the whole setup (Play -> gamemode -> map -> stage) rather
            # than waiting on a teleport that is not coming.
            # Confirmed over two consecutive polls, not one: aborting a live
            # match is expensive, and a single frame caught mid-transition is
            # not worth acting on. Same confirm-then-commit shape wait_wave's
            # target check uses.
            # Once portal cards were displayed, Play is allowed to remain
            # visible behind the result overlay and must not cause a lobby
            # re-entry. The only valid next transition is Select Portal.
            if (not summer_portal_card_state["selected"]
                    and (boss_rush_state is None or boss_rush_state["boss_started"])):
                try:
                    lobby_match = vision.find_image(hwnd, "nav_play")
                except vision.TemplateNotFound:
                    lobby_match = None
                if lobby_match is None:
                    lobby_sightings = 0
                else:
                    lobby_sightings += 1
                    if lobby_sightings >= LOBBY_RESYNC_CONFIRMATIONS:
                        self._log("[Macro] Back on the lobby mid-match (found Play twice) -- "
                                   "re-entering the stage from scratch.")
                        self._set_status(action="Back on the lobby -- re-entering...")
                        return "left"
                    self._log("[Macro] Play button visible mid-match -- confirming on the next poll.")

            afk_clicked_at = self._dismiss_afk_chamber(hwnd, afk_clicked_at)

            if watch_close_popup:
                self._click_close_popup_if_found(hwnd)

            if mode == "expedition":
                # An encounter node parks the client where no result can come
                # from. Checked before the wave result, since the wave check
                # cannot resolve while one is up.
                encounter_state = self._handle_expedition_encounter(
                    hwnd, stop_event, encounter_state)
                result = self._check_expedition_wave_result(hwnd, stop_event)
                if result is not None:
                    return result
                # Every step of the checkpoint chain is individually bounded,
                # but nothing used to notice the whole chain repeating: a
                # Continue that never clears is re-found and re-clicked on
                # every poll, and the run only escaped at
                # MATCH_RESULT_TIMEOUT, half an hour later. Give up once the
                # streak says the checkpoint is not clearing and let the
                # normal recovery path re-enter from the lobby -- exactly
                # what the 30-minute timeout below does, just far sooner.
                stall_detail = None
                if self._expedition_checkpoint_stalled():
                    stall_detail = (
                        f"the same checkpoint has been up for {self._exp_checkpoint_streak} "
                        f"polls without ever clearing (an encounter node waiting on an NPC, "
                        f"or a click that lands visually without registering)")
                elif self._expedition_intercepts_stalled():
                    stall_detail = (
                        f"a popup or reward card has taken every one of the last "
                        f"{self._exp_intercept_streak} polls, so the checkpoint has not been "
                        f"readable that whole time")
                if stall_detail is not None:
                    self._log(
                        f"[Macro] Expedition run hasn't progressed in "
                        f"{EXPEDITION_STALL_TIMEOUT / 60:.0f} min -- {stall_detail}. Abandoning "
                        f"it rather than waiting out the {MATCH_RESULT_TIMEOUT / 60:.0f} min "
                        f"timeout; the task re-enters from the lobby.")
                    self._save_debug_screenshot_unconditional(hwnd, "expedition_no_progress")
                    self._note_checkpoint_cleared()
                    return None
                self._interruptible_sleep(MATCH_RESULT_POLL_INTERVAL, stop_event)
                continue

            if boss_rush_state is None or boss_rush_state["boss_started"]:
                try:
                    victory_match = vision.find_image(hwnd, "victory")
                except vision.TemplateNotFound as exc:
                    self._log(f"[Macro] {exc}")
                    return None
                if victory_match is not None:
                    self._log(f"[Macro] Victory! (score {victory_match['score']:.2f})")
                    return "win"
                try:
                    defeat_match = vision.find_image(hwnd, "defeat")
                except vision.TemplateNotFound as exc:
                    self._log(f"[Macro] {exc}")
                    return None
                if defeat_match is not None:
                    self._log(f"[Macro] Defeat. (score {defeat_match['score']:.2f})")
                    return "loss"
            time.sleep(MATCH_RESULT_POLL_INTERVAL)
        self._log(f'[Macro] Neither "victory" nor "defeat" matched within {MATCH_RESULT_TIMEOUT / 60:.0f} min. '
                   f'If the result screen was actually showing, its reference image isn\'t matching your '
                   f'setup -- add your own crop via Settings > General > Image Manager.')
        self._save_debug_screenshot_unconditional(hwnd, "match_result_timeout")
        return None





    def debug_force_rejoin(self, hwnd, hwnd_getter=None) -> bool:
        """Settings > Debug > "Force Rejoin" -- manually fires the same
        deep-link rejoin _handle_disconnect uses, on demand, with no real
        disconnect needed first. Point of this: resetting Roblox back to a
        known state (the lobby) between test iterations after a code
        change used to mean alt-tabbing over and closing/reopening it by
        hand every time. hwnd_getter is set to the caller's own live
        game_hwnd lookup for the duration of the call (mirrors what a real
        run wires up in _run) so the poll loop can follow Roblox through a
        full relaunch onto whatever new hwnd main.py's own dock watchdog
        re-docks it under, then restored back to whatever it was before --
        there's no run in progress to own this permanently."""
        self._log("[Debug] Forcing a rejoin (deep link)...")
        previous_getter = self._hwnd_getter
        if hwnd_getter is not None:
            self._hwnd_getter = hwnd_getter
        try:
            ok = self._attempt_rejoin(hwnd, threading.Event())
        finally:
            self._hwnd_getter = previous_getter
        self._log(f"[Debug] Force rejoin {'succeeded' if ok else 'failed'}.")
        return ok

    def _save_debug_screenshot_unconditional(self, hwnd, name: str) -> str:
        """A full-window screenshot saved regardless of Settings > Debug >
        "Debug Match Screenshots" -- for failures rare and diagnostically
        useful enough (a long timeout, a menu that never opened) that it's
        worth it unconditionally rather than only when that toggle happened
        to already be on, so a user's bug report actually comes with
        evidence of what was on screen instead of a blind guess. Returns
        the saved path (also attachable to a Discord webhook, see
        _send_event_webhook), or None if it couldn't be saved."""
        try:
            left, top, right, bottom = wm.get_window_rect_screen(hwnd)
            path = vision.save_region_debug(hwnd, name, (0, 0, right - left, bottom - top))
            self._log(f"[Macro] Saved a screenshot for troubleshooting: {path}")
            return path
        except Exception as exc:
            self._log(f"[Macro] Couldn't save a debug screenshot: {exc}")
            return None










    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = int(seconds)
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s" if m else f"{s}s"

    def _select_summer_portal_card(self, hwnd, stop_event: threading.Event, task: dict) -> bool:
        """Detect the displayed portal choices, then click the configured card."""
        # OCR against the whole client is not a safe screen detector: while a
        # battle is active, HUD labels and map text can be misread as the
        # portal banner. Require the dedicated reward-screen crop first. It is
        # editable through Settings > General > Image Manager, so a new game
        # layout can be supported without changing this flow.
        try:
            reward_screen = vision.find_image(hwnd, SUMMER_PORTAL_REWARD_IMAGE)
        except vision.TemplateNotFound:
            reward_screen = None
        if reward_screen is None:
            return False
        coords = getattr(self, "_coords", {})
        card_x = coords.get(
            "portal_card_x",
            coords.get("summer_portal_card_x", SUMMER_PORTAL_CARD_MIDDLE[0]),
        )
        card_y = coords.get(
            "portal_card_y",
            coords.get("summer_portal_card_y", SUMMER_PORTAL_CARD_MIDDLE[1]),
        )
        click_x, click_y = vision.ref_to_screen(hwnd, card_x, card_y)
        retry_round = 0
        while not stop_event.is_set():
            self._set_status(action="Selecting the middle portal card...")
            self._interruptible_sleep(SUMMER_PORTAL_CARD_SETTLE, stop_event)
            retry_round += 1
            for attempt in range(1, 4):
                if self._checkpoint(stop_event):
                    return False
                if not wm.activate_window(hwnd):
                    self._log("[Macro] Couldn't confirm Roblox focus before portal card click.")
                if hasattr(self._mouse, "move_to"):
                    self._mouse.move_to(click_x, click_y)
                time.sleep(0.1)
                self._mouse.click(click_x, click_y)
                self._log(
                    f"[Macro] Portal reward detected -- clicking configured middle card "
                    f"at ({click_x}, {click_y}) (round {retry_round}, attempt {attempt}/3)."
                )
                if attempt < 3:
                    self._interruptible_sleep(0.5, stop_event)
            try:
                result_ready = vision.find_image(hwnd, SUMMER_SELECT_PORTAL_IMAGE) is not None
            except vision.TemplateNotFound:
                result_ready = False
            if result_ready:
                self._log("[Macro] Portal reward accepted -- Select Portal is ready.")
                return True
            try:
                reward_still_visible = vision.find_image(hwnd, SUMMER_PORTAL_REWARD_IMAGE) is not None
            except vision.TemplateNotFound:
                reward_still_visible = False
            if not reward_still_visible:
                self._log("[Macro] Portal reward accepted -- reward screen closed.")
                return True
            self._set_status(action="Portal reward click did not register -- retrying...")
            self._log("[Macro] Portal reward is still visible after 3 clicks -- retrying until it registers.")
            self._interruptible_sleep(1.0, stop_event)
        return False

    @staticmethod
    def _portal_banner_matches(text: str, tier: str, portal: str) -> bool:
        compact = " ".join(text.lower().replace("'", "").split())
        normalized = compact.replace("tier", "tier ")
        normalized = " ".join(normalized.split())
        portal_name = "sky ruins portal" if portal == "Sky Ruins Portal" else "summer portal"
        normalized_compact = normalized.replace(" ", "")
        # Windows OCR may insert punctuation or split the tier digits while
        # the three cards are animating. Require the portal family and accept
        # the tier in either the normal or compact form.
        tier_found = (
            f"tier {tier}" in normalized
            or f"tier{tier}" in normalized
            or f"tler {tier}" in normalized
        )
        portal_found = portal_name in normalized or portal_name.replace(" ", "") in normalized_compact
        return tier_found and portal_found

    def _handle_match_result(self, hwnd, stop_event: threading.Event, task: dict, result: str, duration: str,
                              webhook: dict, repeat: bool) -> bool:
        label = "Victory" if result == "win" else "Defeat"

        # Count a qualifying win (Mastery Story / Challenge) toward the next
        # Auto Crafting pass. This is the single choke point both farms' wins
        # pass through; the call itself no-ops unless crafting is enabled and
        # the win qualifies (see runner_crafting._note_win_for_crafting).
        self._note_win_for_crafting(task, result)

        # A Battle-phase quick-place chain (see _run_place_unit_block) could
        # still be holding Shift down right up to the moment Victory/Defeat
        # actually landed -- released here, before anything else, so it's
        # never still held into the result screen and beyond.
        self._release_quick_place_shift()

        # The reward row streams its items in one at a time, so give it a beat
        # to finish before either the screenshot or the relic check reads the
        # screen (see RESULT_CAPTURE_DELAY) -- otherwise both can catch a
        # half-populated row.
        time.sleep(RESULT_CAPTURE_DELAY)

        # Only the CAPTURE (one screenshot) has to happen while the result
        # screen is actually still up -- the run's result is now reported by
        # SENDING that image to the webhook (plus a rendered win/loss card),
        # not by OCRing the stats/reward panels off the screen, so there's
        # just the one grab left here and the send itself runs on its own
        # thread instead of holding up Repeat/Leave Stage. Only captured when
        # a webhook is actually configured to receive it.
        self._set_status(action=f"Capturing {label} screen...")
        webhook_wants_shot = bool(webhook and webhook.get("enabled") and webhook.get("url"))
        result_screenshot = self._capture_result_screenshot(hwnd) if webhook_wants_shot else None

        map_name = task.get("map") or "-"
        webhook_done = threading.Event()
        threading.Thread(
            target=self._finish_match_result_background,
            args=(result, map_name, duration, task, webhook, result_screenshot, webhook_done),
            daemon=True,
        ).start()
        self._log(f"[Macro] {label} ({duration}) -- reporting in the background.")

        # The cursor is moved to the same near-empty corner
        # _reset_unit_info_panel uses first, so a leftover hover
        # state/tooltip from whatever was under the cursor can't throw off
        # whichever button gets clicked next.
        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        self._mouse.move_to(left + self._coords["unit_info_reset_x"], top + self._coords["unit_info_reset_y"])
        time.sleep(0.1)
        if task.get("mode") == "summer" and task.get("summer_mode") == "Portal Mode":
            self._set_status(action="Waiting for webhook sent...")
            webhook_done.wait(timeout=WEBHOOK_SEND_TIMEOUT)

        # A level-up reward-card modal can land exactly as the match ends
        # (confirmed from a real stuck report: "Couldn't find repeat_stage"
        # after the FULL 8s search window, twice, both times ~9s after
        # Extracted -- consistent with something blocking the result panel
        # the whole time, not just a slow render), before the Repeat/Leave
        # Stage panel has even rendered. Dismissed here too, not just
        # mid-battle, looping until it's actually gone (it can re-show
        # between multiple level-ups) or REWARD_CARD_CLEAR_TIMEOUT runs out.
        card_deadline = time.time() + REWARD_CARD_CLEAR_TIMEOUT
        dismissed_a_card = False
        while self._dismiss_reward_card_if_found(hwnd) and time.time() < card_deadline:
            dismissed_a_card = True
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(0.5)
        if dismissed_a_card:
            # The dismiss click above lands dead center of the screen --
            # right where an item/reward tooltip hovers in and covers the
            # Repeat/Leave Stage buttons (reported from real testing). Reset
            # back to the same near-empty corner used above before this
            # loop ever ran, so that hover state doesn't linger into the
            # repeat_stage/leave_stage search below.
            self._mouse.move_to(left + self._coords["unit_info_reset_x"], top + self._coords["unit_info_reset_y"])
            time.sleep(0.1)

        if result == "win" and not self._clear_result_obtainment_modal(hwnd, stop_event):
            return False

        if (result == "win" and repeat and task.get("mode") == "summer"
                and task.get("summer_mode") == "Portal Mode"):
            return self._select_summer_portal_for_repeat(hwnd, stop_event, task)

        # Matchmaking never uses Repeat Stage, even with more repeats left --
        # a matchmade lobby is a one-shot party for that specific match, not
        # something "repeat the same stage" can just re-queue into the way
        # it can for Story/Solo, so every matchmaking repeat has to leave
        # and go through Enter Matchmaking again from the lobby (see
        # _run_task's repeat loop, which re-runs _run_task_setup instead of
        # _wait_teleport_in whenever this is why it's about to see Leave
        # Stage clicked with more repeats still left).
        is_matchmaking = task.get("play_mode") == "matchmaking"
        if repeat and not is_matchmaking:
            # More repeats left on this task -- Repeat Stage re-queues the
            # same stage directly, skipping the lobby/gamemode/map/stage
            # picks entirely (see _run_task_setup, which only runs once per
            # task, not once per repeat).
            if task.get("mode") == "tower":
                repeat_image = "Next_Floor" if result == "win" else "Repeat_Floor"
                repeat_label = repeat_image
            else:
                repeat_image = "repeat_stage"
                repeat_label = "Repeat Stage"
            self._set_status(action=f"{label} -- clicking {repeat_label}...")
            if not self._click_and_verify_gone(hwnd, stop_event, repeat_image, NAV_CLICK_TIMEOUT):
                self._log(f'[Macro] "{repeat_label}" not found -- can\'t continue this task\'s repeats, stopping.')
                return False
            # _click_and_verify_gone only confirms the repeat_stage BUTTON
            # image is gone, not that the whole Victory/Defeat results panel
            # actually closed -- confirmed from a real capture: the button
            # itself can visually change/disappear (so the check above
            # reports success) while the full result modal is still up on
            # screen behind it, and Pre Start went on to place units right
            # through/behind it. The banner ribbon is the more reliable
            # "actually closed" signal, so wait for THAT to clear too before
            # treating the repeat as ready to continue into. Best-effort --
            # a still-showing banner after the timeout just gets logged, not
            # treated as a hard failure, since the repeat may have gone
            # through fine underneath anyway.
            self._wait_for_image_gone(hwnd, (label.lower(),), REPEAT_STAGE_MODAL_CLEAR_TIMEOUT, stop_event)
            return True

        # Last repeat of this task (or the whole queue) -- back out to the
        # lobby so the next task's setup (or a clean stop) starts from a
        # known state instead of sitting on the result screen. Verified/
        # retried, not a one-shot click -- a dropped click here used to
        # just leave the run sitting on the result screen forever with
        # nothing else ever noticing or retrying.
        self._set_status(action=f"{label} -- clicking Leave Stage...")
        if not self._click_and_verify_gone(
                hwnd, stop_event, "leave_stage", NAV_CLICK_TIMEOUT, success_name="return"):
            self._log('[Macro] "Leave Stage" not found -- stopping.')
            return False
        self._click_return_to_lobby_if_found(hwnd, stop_event)
        return True

    def _select_summer_portal_for_repeat(self, hwnd, stop_event: threading.Event, task: dict) -> bool:
        """Open Select Portal and choose the task's configured portal tier."""
        portal = task.get("map") or "Summer Portal"
        tier = str(task.get("portal_tier") or "1")
        if not SUMMER_PORTAL_IMAGES.get((portal, tier)):
            self._log(f'[Macro] Unknown Summer portal "{portal}" tier "{tier}".')
            return False
        self._set_status(action="Selecting the next Summer portal...")
        # A portal repeat must never fall through to task recovery: recovery
        # returns to the lobby and starts the first-run nav_event flow again.
        # Keep polling the current result/selection screen until the configured
        # portal is selected and Start Game is visible.
        if self._checkpoint(stop_event):
            return False
        opened_selection = (
            self._click_found_image(
                hwnd, SUMMER_SELECT_PORTAL_IMAGE, NAV_CLICK_TIMEOUT, stop_event) is not None
            or self._select_summer_portal_from_grid(hwnd, stop_event, portal, tier)
        )
        if not opened_selection:
            self._log('[Macro] Select Portal/Portal Selection did not appear.')
            return False

        selected = self._select_summer_portal_from_grid(hwnd, stop_event, portal, tier)
        if not selected:
            self._log('[Macro] Portal Selection did not show the configured portal.')
            return False

        selected_button = (
            self._click_found_image(
                hwnd, SUMMER_PORTAL_SELECTION_SELECT_IMAGE,
                SUMMER_SCREEN_TIMEOUT,
                stop_event) is not None
        )
        if not selected_button:
            self._log('[Macro] Portal selection "Select" button did not appear.')
            return False

        started = (
            self._click_found_image(
                hwnd, "nav_start_game",
                SUMMER_SCREEN_TIMEOUT,
                stop_event) is not None
        )
        if not started:
            self._log('[Macro] "Start Game" did not appear after selecting the portal.')
            return False
        return not self._checkpoint(stop_event)

    def _select_summer_portal_from_grid(self, hwnd, stop_event: threading.Event,
                                        portal: str, tier: str) -> bool:
        """Select the configured portal card using the initial inventory flow."""
        image_name = SUMMER_PORTAL_IMAGES[(portal, tier)][0]
        # Use color matching first: portal tiers share the same silhouette and
        # label, while their border/background color identifies the tier. The
        # selection inventory can be positioned differently from the first-run
        # inventory. Keep the family row isolated so a visually similar card
        # from the other portal family cannot win the full-screen match.
        grid_region = None
        if tier != "Secret":
            grid_region = (220, 175, 450, 135) if portal == "Sky Ruins Portal" else (220, 275, 450, 125)
        color_click = getattr(self, "_click_found_color_image", None)
        match = (
            color_click(hwnd, image_name, SUMMER_SCREEN_TIMEOUT, stop_event,
                        region=grid_region, threshold=0.60, log_failure=False)
            if color_click is not None else None
        )
        if match is None:
            # Keep a grayscale fallback for older/custom portal templates.
            match = self._click_found_image(
                hwnd, image_name, SUMMER_SCREEN_TIMEOUT, stop_event,
                threshold=0.60, region=grid_region)
        if match is None:
            self._log(f'[Macro] Portal card "{image_name}" was not found in Portal Selection.')
            return False
        return True

    def _click_found_color_image(self, hwnd, name: str, timeout: float,
                                 stop_event: threading.Event = None,
                                 region: tuple = None, threshold: float = 0.90,
                                 log_failure: bool = True) -> dict:
        try:
            match = vision.wait_for_color_image(
                hwnd, name, region=region, threshold=threshold,
                timeout=timeout, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return None
        if match is None:
            if log_failure and (stop_event is None or not stop_event.is_set()):
                self._log(f'[Macro] Color image "{name}" not found within {timeout:.0f}s -- stopping.')
            return None
        self._log(f'[Macro] Found color image "{name}" (score {match["score"]:.2f}) -- clicking it.')
        vision.click_match(self._mouse, hwnd, match)
        return match

    def _finish_match_result_background(self, result: str, map_name: str, duration: str, task: dict,
                                          webhook: dict, result_screenshot: str = None,
                                          webhook_done: threading.Event = None) -> None:
        self._record_result(result, map_name, duration)
        try:
            self._send_result_webhook(webhook, result, task, duration, result_screenshot)
        finally:
            # A temp file per match -- deleted once it's been sent (or the
            # send was skipped), whatever the outcome, so they don't pile up.
            if result_screenshot:
                try:
                    os.remove(result_screenshot)
                except OSError:
                    pass
            if webhook_done is not None:
                webhook_done.set()

    def _capture_result_screenshot(self, hwnd) -> str:
        """Full-window screenshot of the Victory/Defeat screen, saved to a
        temp PNG for the match-result webhook to attach. Captured
        synchronously while the result screen is still up (the send itself,
        which composites the win/loss card under this and uploads it, then
        runs on the background thread, which deletes this file afterward).
        Returns the path, or None if it couldn't be captured."""
        try:
            import tempfile
            fd, path = tempfile.mkstemp(prefix="ae_result_", suffix=".png")
            os.close(fd)
            saved = vision.save_window_screenshot(hwnd, path)
            if not saved:
                try:
                    os.remove(path)
                except OSError:
                    pass
                return None
            return saved
        except Exception as exc:
            self._log(f"[Macro] Couldn't capture the result screenshot for the webhook: {exc}")
            return None

    @staticmethod
    def _tree_rows(rows) -> str:
        """Tree-style lines for one inline embed field (the field's own name
        is the header, so no bracketed header here): each (label, value) as a
        box-drawing branch with a bold label and a code value, last row on
        the └ corner. e.g. ├ **Elapsed:** `11h 27m` / └ ..."""
        rows = [r for r in rows if r is not None]
        lines = []
        for i, (label, value) in enumerate(rows):
            branch = "└" if i == len(rows) - 1 else "├"
            lines.append(f"{branch} **{label}:** `{value}`")
        return "\n".join(lines)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """Session runtime as "11h 27m" / "27m 4s" / "45s" -- longer scale
        than _format_duration (a single match), so it leads with hours."""
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _send_result_webhook(self, webhook: dict, result: str, task: dict, duration: str,
                              screenshot_path: str = None) -> None:
        url = (webhook or {}).get("url")
        if not url or not webhook.get("enabled"):
            return
        from . import webhook as webhook_module

        is_win = result == "win"
        map_name = task.get("map") or "-"
        mode = task.get("mode") or "story"
        raw_stage = task.get("stage") or "-"
        # Raid/Event pick Acts; Story's Infinite/Mastery are named; the rest
        # are numbered stages.
        if mode == "tower":
            stage = "-"
        elif mode in ("raid", "event"):
            stage = f"Act {raw_stage}" if raw_stage != "-" else "-"
        elif str(raw_stage).isdigit():
            stage = f"Stage {raw_stage}"
        else:
            stage = str(raw_stage)
        # Raid and Infinite/Mastery stages are locked to Hard in-game (see
        # _run_task_setup's identical check) -- the task's own difficulty
        # setting never actually applies there, so reporting it verbatim
        # was showing e.g. "Normal" for a run that was really Hard. Event has
        # no difficulty at all.
        if (mode == "raid" or raw_stage in SPECIAL_STAGES_NO_DIFFICULTY
                or (mode == "story" and task.get("story_event") not in (None, "", "Normal"))):
            difficulty = "Hard"
        elif mode in ("event", "tournament", "tower"):
            difficulty = "-"
        else:
            difficulty = task.get("difficulty") or "-"

        # Fresh session/all-time snapshot -- already includes THIS match, since
        # the webhook is sent right after _record_result bumped the counters.
        try:
            snap = self._get_run_stats() or {}
        except Exception:
            snap = {}
        sw, sl = snap.get("session_wins", 0), snap.get("session_losses", 0)
        aw, al = snap.get("all_time_wins", 0), snap.get("all_time_losses", 0)
        session_time = self._format_elapsed(time.time() - snap["session_start"]) if snap.get("session_start") else "-"

        where = " · ".join(p for p in (map_name, stage, difficulty) if p and p != "-")
        session_rate = f"{round(sw / (sw + sl) * 100)}%" if (sw + sl) else "-"
        all_time_rate = f"{round(aw / (aw + al) * 100)}%" if (aw + al) else "-"
        tuc = snap.get("time_until_challenge", "-")
        runs_per_hour = snap.get("runs_per_hour", "-")

        # Inline, tree-style fields (bold labels, code values) side by side --
        # Match / Session / All Time. The visual win/loss tiles + activity
        # grid live in the separate status-card image below.
        fields = [
            {"name": "⚔️ Match", "value": self._tree_rows([
                ("Result", "Victory \U0001F3C6" if is_win else "Defeat \U0001F480"),
                ("Duration", duration or "-"),
            ] + ([("Stage", where)] if where else [])), "inline": True},
            {"name": "\U0001F4CA Session", "value": self._tree_rows([
                ("Elapsed", session_time),
                ("Record", f"{sw}W · {sl}L"),
                ("Rate", session_rate),
                ("Runs/h", runs_per_hour),
                ("Challenge", tuc),
            ]), "inline": True},
            {"name": "\U0001F3C6 All Time", "value": self._tree_rows([
                ("Record", f"{aw}W · {al}L"),
                ("Rate", all_time_rate),
            ]), "inline": False},
        ]
        # A plain incoming webhook can't render real buttons (those need a
        # bot/app-owned webhook) -- so the links go here as clickable masked
        # markdown links, which render everywhere. Full width, at the bottom.
        links = (f"[\U0001F4AC Discord]({DISCORD_INVITE_URL})   •   "
                 f"[\U0001F4FA YouTube]({YOUTUBE_URL})   •   "
                 f"[\U00002B50 GitHub]({GITHUB_REPO_URL})")
        fields.append({"name": "\U0001F517 Links", "value": links, "inline": False})
        result_word = "Victory" if is_win else "Defeat"
        description = (f"{result_word} on **{where}** — session match **#{sw + sl}**."
                       if where else f"{result_word} — session match **#{sw + sl}**.")

        version = snap.get("version")
        footer = "VFERDZ MACRO | Anime Expeditions" + (f" · v{version}" if version else "")
        main_embed = {
            "title": "Victory! \U0001F3C6" if is_win else "Defeat \U0001F480",
            "color": 0x3FBF6F if is_win else 0xE05A6D,
            "description": description,
            "fields": fields,
            "footer": {"text": footer},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mention_id = (webhook or {}).get("mention_id")
        content = f"<@{mention_id}>" if mention_id else ""
        silent = bool(webhook.get("silent"))

        # Two SEPARATE images (each in its own embed, so Discord shows them as
        # distinct pictures rather than one stacked image): the rendered
        # status card, and the live result screenshot. Best-effort -- a failed
        # render/read just drops that one image, never the whole notification.
        embeds = [main_embed]
        files = []
        try:
            import cv2
            from . import status_card
            card = status_card.render_status_card_bgr(
                is_win=is_win, action="Match Finished", last_run_duration=duration,
                last_run_win=is_win, time_until_challenge=tuc,
                session_wins=sw, session_losses=sl, all_time_wins=aw, all_time_losses=al,
                runs_per_hour=runs_per_hour, results=snap.get("results"))
            if card is not None:
                ok, buf = cv2.imencode(".png", card)
                if ok:
                    files.append(("card.png", buf.tobytes()))
                    main_embed["image"] = {"url": "attachment://card.png"}
        except Exception as exc:
            self._log(f"[Macro] Couldn't render the status card: {exc}")
        if screenshot_path and os.path.isfile(screenshot_path):
            try:
                with open(screenshot_path, "rb") as f:
                    files.append(("shot.png", f.read()))
                embeds.append({"color": main_embed["color"], "image": {"url": "attachment://shot.png"}})
            except OSError:
                pass

        try:
            send_result = webhook_module.send_rich(
                url, embeds=embeds, file_attachments=files,
                content=content, silent=silent)
        except Exception as exc:
            self._log(f"[Macro] Webhook send failed: {exc}")
            return
        if send_result["ok"]:
            self._log("[Macro] Webhook sent.")
        else:
            self._log(f"[Macro] Webhook send failed: {send_result['reason']}")

    def _run_prestart(self, hwnd, stop_event: threading.Event, task: dict, default_walk_paths: dict,
                        first_repeat: bool = True, run_blocks: bool = True) -> bool:
        # Camera setup runs ONCE per fresh entry into a stage (same
        # first_repeat gate as Team Loadout and the Walk Path block below)
        # -- it used to re-run on every repeat as a "per-match reset", but
        # the camera actually holds its angle across a repeat's
        # re-teleport, so re-running just re-dragged an already-correct
        # camera off its spot every loop.
        #
        # Expedition gets its own sequence: the standard drag-down + O
        # zoom-hold doesn't frame Expedition maps right, so it uses the
        # drag-down + Left-arrow rotate instead (the same sequence Settings
        # > Debug > Camera Setup 3 tests) -- 730ms rotate, then a short O
        # tap for a small zoom step (duration user-tunable: Settings >
        # Debug > "Expedition Camera Zoom", 100ms default).
        if first_repeat:
            self._log("[Macro] Pre Start: setting up the camera...")
            self._set_status(action="Setting up camera...")
            # nav_unitmanager (just confirmed by _wait_teleport_in) is a HUD
            # element and can render a beat before the character/camera
            # controller has actually finished attaching to the freshly-
            # spawned avatar -- this blind right-click-drag has no visual
            # confirmation of its own to wait on, so a short settle here is
            # what catches that rare case instead of dragging on a camera
            # that isn't ready to receive it yet.
            self._interruptible_sleep(CAMERA_SETUP_SETTLE, stop_event)
            if self._checkpoint(stop_event):
                return False
            try:
                if task.get("mode") == "expedition":
                    camera.run_camera_drag_hold(self._mouse, self._keyboard, hwnd, hold_ms=730,
                                                 o_tap_ms=self._expedition_camera_o_ms)
                else:
                    camera.run_camera_setup(self._mouse, self._keyboard, hwnd)
                self._log("[Macro] Camera setup done.")
            except Exception as exc:
                self._log(f"[Macro] Camera setup failed: {exc}")
        else:
            # Skipping the camera work also skips the only thing standing
            # between teleporting in and placing units, so the settle it
            # relied on has to be kept explicitly -- see REPEAT_ENTRY_SETTLE.
            self._log("[Macro] Repeat of the same stage -- skipping camera setup (already set on entry). "
                      f"Letting the map settle for {REPEAT_ENTRY_SETTLE:.0f}s before placing.")
            self._interruptible_sleep(REPEAT_ENTRY_SETTLE, stop_event)
        if self._checkpoint(stop_event):
            return False

        # Team Loadout only makes sense the FIRST time a task enters a stage --
        # it sets up units/equipment before the match even starts, so
        # re-applying it on every repeat (this used to run unconditionally,
        # same bug the walk below already had a fix for) was pointlessly
        # re-pressing H and re-picking a loadout mid-repeat-cycle, which
        # could land on the wrong screen entirely if the panel wasn't in the
        # exact state it expects and produce exactly the kind of "it bugs
        # out" behavior this was reported as.
        if first_repeat:
            team_loadout = self._team_loadout_key(task)
            if team_loadout and team_loadout == self._last_applied_team_loadout:
                self._log(f"[Macro] Reusing Team Loadout {team_loadout[0]} "
                          f"(equipment: {team_loadout[1]}) from the previous task.")
            else:
                if team_loadout and team_loadout != self._last_applied_team_loadout:
                    # Do not let a failed attempt for a different team leave
                    # an older successful key eligible for a later task.
                    self._last_applied_team_loadout = None
                if not self._apply_team_loadout(hwnd, stop_event, task):
                    if stop_event.is_set():
                        return False
                    self._log("[Macro] Team Loadout didn't actually apply -- failing this match setup so it "
                              "retries from the lobby instead of starting a round with no team equipped.")
                    return False
        else:
            self._log("[Macro] Repeat of the same stage -- skipping Team Loadout (already applied on entry).")
        if self._checkpoint(stop_event):
            return False

        if not run_blocks:
            self._log("[Boss Rush] Deferring Pre Start blocks until after Start Game and the gate route.")
            return True

        # Walk Path used to be a fixed step here, always running before any
        # of the template's own blocks and never reorderable -- now it's a
        # real block within the list itself (see _run_prestart_blocks'
        # "walk_path" handling / _run_walk_path_block), so it runs wherever
        # it's actually positioned relative to Setting/Place Unit blocks
        # instead of always jumping the whole list.
        self._run_prestart_blocks(hwnd, stop_event, task, first_repeat, default_walk_paths)
        if self._checkpoint(stop_event):
            return False
        return True

    def _team_loadout_key(self, task: dict):
        """Return the normalized team/equipment pair configured by a task.

        Invalid or legacy templates return ``None`` because they do not have
        a loadout that can safely be reused. The actual application path keeps
        its existing logging and validation behavior.
        """
        macro_name = task.get("macro")
        if not macro_name:
            return None
        from . import templates as tpl
        data = tpl.load_template(macro_name)
        blocks = data.get("blocks") or {}
        if not isinstance(blocks, dict):
            return None
        team = blocks.get("team") or ""
        try:
            team_num = int(team)
        except (TypeError, ValueError):
            return None
        if not (1 <= team_num <= TEAM_LOADOUT_MAX_SUPPORTED):
            return None
        equipment = blocks.get("equipment") if blocks.get("equipment") in ("include", "exclude") else "include"
        return team_num, equipment

    def _apply_team_loadout(self, hwnd, stop_event: threading.Event, task: dict,
                            team_override=None) -> bool:
        """Open Unit Teams, select one of its eight rows, set saved equipment,
        and use the selected team.

        No team configured, an unrecognized/out-of-range slot number (a
        template config problem retrying can't fix either) still just skip
        with a log line and report success -- there was never a team to
        apply. Once a team number is valid, selection failures report false
        so a real run retries instead of starting with the wrong team.
        """
        macro_name = task.get("macro")
        if not macro_name and team_override is None:
            return True
        from . import templates as tpl
        data = tpl.load_template(macro_name) if macro_name else {}
        blocks = data.get("blocks") or {}
        if isinstance(blocks, list):
            if team_override is None:
                return True  # old-format template -- same as _run_prestart_blocks
            blocks = {}
        raw_team = (
            team_override if team_override is not None
            else blocks.get("team") if blocks.get("team") not in (None, "")
            else task.get("team")
        )
        team = str(raw_team).strip() if raw_team is not None else ""
        if not team:
            return True
        equipment = blocks.get("equipment")
        if equipment not in ("include", "exclude"):
            equipment = task.get("equipment") if task.get("equipment") in ("include", "exclude") else "include"

        try:
            team_num = int(team)
        except (TypeError, ValueError):
            self._log(f'[Macro] Team Loadout "{team}" isn\'t a recognized slot number -- skipping.')
            return True
        if not (1 <= team_num <= TEAM_LOADOUT_MAX_SUPPORTED):
            self._log(f'[Macro] Team Loadout {team_num} is outside the supported range 1-'
                       f'{TEAM_LOADOUT_MAX_SUPPORTED} -- skipping.')
            return True

        team_loadout = (team_num, equipment)
        # A real attempt for a different configuration makes the old success
        # unusable, even if this attempt later fails.
        self._last_applied_team_loadout = None
        self._log(f"[Macro] Applying Team Loadout {team_num} (equipment: {equipment})...")
        self._set_status(action=f"Applying Team Loadout {team_num}...")
        self._keyboard.tap(ord("H"))

        try:
            team_match = vision.wait_for_image(hwnd, "team", timeout=TEAM_PANEL_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't confirm the team panel opened: {exc}")
            return False
        if team_match is None:
            if not stop_event.is_set():
                self._log(f'[Macro] "team" not found within {TEAM_PANEL_TIMEOUT:.0f}s after pressing H -- '
                           f'the team panel never opened. Team Loadout {team_num} was NOT applied.')
            return False
        applied = False
        team_panel_applied = False
        try:
            team_panel_applied = self._apply_team_loadout_panel(
                hwnd, stop_event, team_match, team_num, equipment)
            if not team_panel_applied:
                return False
            if not self._close_unit_manager_panel(hwnd, stop_event):
                return False
            applied = True
            if applied:
                self._last_applied_team_loadout = team_loadout
            return applied
        finally:
            # On failure, H closes the panel so the next attempt doesn't
            # inherit its focus.
            if not team_panel_applied:
                self._keyboard.tap(ord("H"))
                self._log("[Macro] Closed the Team Loadout panel after a failed selection.")
                self._interruptible_sleep(1.0, stop_event)

    def _close_unit_manager_panel(self, hwnd, stop_event: threading.Event) -> bool:
        self._interruptible_sleep(TEAM_LOADOUT_ACTION_SETTLE, stop_event)
        for attempt in range(1, TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS + 1):
            if self._checkpoint(stop_event):
                return False
            try:
                panel_match = vision.find_image(hwnd, "team")
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] Can't verify whether the Units panel is open: {exc}")
                self._save_debug_screenshot_unconditional(hwnd, "team_loadout_units_close_failed")
                return False
            if panel_match is None:
                self._log("[Macro] Units panel is closed.")
                return True

            self._keyboard.tap(ord("H"))
            self._log(f"[Macro] Pressed H to close the Units panel (attempt "
                      f"{attempt}/{TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS}); waiting for it to close.")
            deadline = time.time() + TEAM_LOADOUT_CLOSE_VERIFY_TIMEOUT
            while time.time() < deadline:
                if self._checkpoint(stop_event):
                    return False
                try:
                    panel_match = vision.find_image(hwnd, "team")
                except vision.TemplateNotFound as exc:
                    self._log(f"[Macro] Can't verify the Units panel closed: {exc}")
                    self._save_debug_screenshot_unconditional(hwnd, "team_loadout_units_close_failed")
                    return False
                if panel_match is None:
                    self._log("[Macro] Units panel closed.")
                    return True
                time.sleep(0.2)
            if attempt < TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS:
                self._log("[Macro] Units panel is still visible -- retrying H.")

        self._log(f"[Macro] Units panel stayed open after "
                  f"{TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS} close attempts.")
        self._save_debug_screenshot_unconditional(hwnd, "team_loadout_units_close_failed")
        return False

    def _apply_team_loadout_panel(self, hwnd, stop_event: threading.Event, team_match, team_num: int,
                                    equipment: str) -> bool:
        # The "team" image proves only that the Unit Manager is open. Clicking
        # it opens the updated Unit Teams screen. The
        # old path waited a blind 0.3s and immediately scrolled/clicked; if
        # this click was early or dropped, every later action landed against
        # the wrong screen. Verify the destination's own title and retry THIS
        # click before touching any loadout row.
        manual_team_click = self._optional_cxy("team_button")
        if manual_team_click is not None:
            self._log(f"[Macro] Using manual Teams click point ({manual_team_click[0]}, "
                       f"{manual_team_click[1]}).")
        loadout_open = None
        for attempt in range(1, TEAM_LOADOUT_OPEN_RETRY_ATTEMPTS + 1):
            if self._checkpoint(stop_event):
                return False
            if attempt > 1:
                self._log(f'[Macro] Unit Teams did not open -- retrying the Teams button '
                           f'(attempt {attempt}/{TEAM_LOADOUT_OPEN_RETRY_ATTEMPTS}).')
            if manual_team_click is None:
                vision.click_match(self._mouse, hwnd, team_match)
            else:
                vision.click_match(self._mouse, hwnd, {
                    "cx": manual_team_click[0], "cy": manual_team_click[1],
                })
            self._log("[Macro] Clicked Teams -- waiting for Unit Teams.")
            try:
                loadout_open = vision.wait_for_image(
                    hwnd, "team_loadout_open", threshold=TEAM_LOADOUT_OPEN_THRESHOLD,
                    timeout=TEAM_PANEL_TIMEOUT, stop_event=stop_event)
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] Can't verify the Unit Teams screen: {exc}")
                return False
            if loadout_open is not None:
                break
            if stop_event.is_set():
                return False
        if loadout_open is None:
            self._log(f'[Macro] Unit Teams screen never opened after '
                      f'{TEAM_LOADOUT_OPEN_RETRY_ATTEMPTS} Teams clicks.')
            self._save_debug_screenshot_unconditional(hwnd, "team_loadout_open_failed")
            return False
        self._log(f'[Macro] Unit Teams screen open (score {loadout_open["score"]:.2f}).')
        # The title arrives before the row animation has completely settled.
        time.sleep(TEAM_LOADOUT_OPEN_SETTLE)
        if self._checkpoint(stop_event):
            return False

        row_x, first_row_y = self._cxy("team_loadout")
        row_y = first_row_y + (team_num - 1) * int(self._coords["team_loadout_row_height"])
        row_screen_x, row_screen_y = vision.ref_to_screen(hwnd, row_x, row_y)
        team_selected = False
        for attempt in range(1, TEAM_LOADOUT_SELECT_RETRY_ATTEMPTS + 1):
            if self._checkpoint(stop_event):
                return False
            self._mouse.click(row_screen_x, row_screen_y)
            self._log(f"[Macro] Clicked Team {team_num} (attempt "
                      f"{attempt}/{TEAM_LOADOUT_SELECT_RETRY_ATTEMPTS}); waiting for selection.")
            deadline = time.time() + TEAM_LOADOUT_SELECT_VERIFY_TIMEOUT
            while time.time() < deadline:
                if self._checkpoint(stop_event):
                    return False
                frame = vision.capture_game_bgr(hwnd)
                selected = _team_loadout_row_selected(frame, row_x, row_y)
                if selected is True:
                    team_selected = True
                    break
                time.sleep(0.2)
            if team_selected:
                break
            if attempt < TEAM_LOADOUT_SELECT_RETRY_ATTEMPTS:
                self._log(f"[Macro] Team {team_num} is not visibly selected -- retrying its row.")
        if not team_selected:
            self._log(f"[Macro] Team {team_num} selection was not detected after "
                      f"{TEAM_LOADOUT_SELECT_RETRY_ATTEMPTS} clicks.")
            self._save_debug_screenshot_unconditional(hwnd, "team_loadout_select_failed")
            return False

        equipment_x, equipment_y = self._cxy("team_equipment")
        frame = vision.capture_game_bgr(hwnd)
        equipment_checked = _team_equipment_checkbox_checked(frame, equipment_x, equipment_y)
        if equipment_checked is None:
            self._log("[Macro] Couldn't read the saved-equipment checkbox; Team Loadout was NOT applied.")
            self._save_debug_screenshot_unconditional(hwnd, "team_loadout_equipment_state_failed")
            return False
        should_check = equipment == "include"
        if equipment_checked != should_check:
            equipment_screen_x, equipment_screen_y = vision.ref_to_screen(
                hwnd, equipment_x, equipment_y)
            self._mouse.click(equipment_screen_x, equipment_screen_y)
            time.sleep(TEAM_LOADOUT_ACTION_SETTLE)
            frame = vision.capture_game_bgr(hwnd)
            equipment_checked = _team_equipment_checkbox_checked(frame, equipment_x, equipment_y)
            if equipment_checked != should_check:
                self._log(f'[Macro] Could not set saved equipment to "{equipment}".')
                self._save_debug_screenshot_unconditional(hwnd, "team_loadout_equipment_state_failed")
                return False
        self._log(f"[Macro] Saved equipment: {equipment}.")
        if self._checkpoint(stop_event):
            return False

        time.sleep(TEAM_LOADOUT_ACTION_SETTLE)
        use_x, use_y = self._cxy("team_use")
        use_screen_x, use_screen_y = vision.ref_to_screen(hwnd, use_x, use_y)
        try:
            use_match = vision.find_image(hwnd, "team_use")
        except vision.TemplateNotFound:
            use_match = None
            self._log('[Macro] Image Manager has no "team_use" image; using the saved Use Team coordinates.')
        if self._checkpoint(stop_event):
            return False
        if use_match is None:
            self._mouse.click(use_screen_x, use_screen_y)
        else:
            vision.click_match(self._mouse, hwnd, use_match)
        self._log("[Macro] Clicked Use Team; waiting before closing Unit Teams.")
        self._interruptible_sleep(TEAM_LOADOUT_ACTION_SETTLE, stop_event)

        panel_closed = False
        for attempt in range(1, TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS + 1):
            if self._checkpoint(stop_event):
                return False
            try:
                close_match = vision.find_image(hwnd, "nav_closeui")
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] Can't find the nav_closeui Image Manager template: {exc}")
                self._save_debug_screenshot_unconditional(hwnd, "team_loadout_close_failed")
                return False
            if close_match is None:
                self._log("[Macro] Couldn't detect the Unit Teams close button with nav_closeui.")
                self._save_debug_screenshot_unconditional(hwnd, "team_loadout_close_failed")
                return False
            vision.click_match(self._mouse, hwnd, close_match)
            self._log(f"[Macro] Clicked Unit Teams close button (attempt "
                      f"{attempt}/{TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS}); waiting for the panel to close.")

            deadline = time.time() + TEAM_LOADOUT_CLOSE_VERIFY_TIMEOUT
            while time.time() < deadline:
                if self._checkpoint(stop_event):
                    return False
                try:
                    panel_match = vision.find_image(hwnd, "team_loadout_open")
                except vision.TemplateNotFound as exc:
                    self._log(f"[Macro] Can't verify Unit Teams closed: {exc}")
                    self._save_debug_screenshot_unconditional(hwnd, "team_loadout_close_failed")
                    return False
                if panel_match is None:
                    panel_closed = True
                    break
                time.sleep(0.2)
            if panel_closed:
                break
            if attempt < TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS:
                self._log("[Macro] Unit Teams is still visible -- retrying its close button.")
        if not panel_closed:
            self._log(f"[Macro] Unit Teams stayed open after "
                      f"{TEAM_LOADOUT_CLOSE_RETRY_ATTEMPTS} close-button clicks; "
                      f"Team {team_num} was NOT confirmed.")
            self._save_debug_screenshot_unconditional(hwnd, "team_loadout_close_failed")
            return False
        return True











    def _wait_teleport_in(self, hwnd, stop_event: threading.Event, webhook: dict = None,
                            task: dict = None, timeout: float = None) -> bool:
        # nav_unitmanager only renders once you're actually in the match (not
        # during the loading/teleport transition), so waiting for it is the
        # confirmation the teleport actually finished.
        timeout = TELEPORT_IN_TIMEOUT if timeout is None else timeout
        self._log(f'[Macro] Waiting to teleport in-game (watching for "nav_unitmanager", up to '
                   f'{timeout:.0f}s)...')
        self._set_status(action='Waiting to teleport in-game ("nav_unitmanager")...')
        portal_repeat = (
            task is not None
            and task.get("mode") == "summer"
            and task.get("summer_mode") == "Portal Mode"
        )
        result = self._wait_for_teleport_result(
            hwnd, stop_event, timeout,
            lobby_grace=PORTAL_REPEAT_LOBBY_GRACE if portal_repeat else 0.0,
        )
        if result == "ok":
            self._log("[Macro] Teleported in-game.")
            return True
        if result == "disconnected":
            self._handle_disconnect(hwnd, stop_event, webhook, task)
            return False
        if result == "lobby":
            # Back on the lobby with no teleport coming. Reported as a failed
            # setup so _run_task recovers and retries the task from the top,
            # rather than sitting out the remaining timeout.
            self._log("[Macro] Still on the lobby -- no teleport is coming (matchmaking left or never "
                       "started). Retrying this task from the lobby.")
            self._set_status(action="Back on the lobby -- retrying the task...")
            return False
        if result == "timeout" and not stop_event.is_set():
            screenshot_path = self._save_debug_screenshot_unconditional(hwnd, "teleport_timeout")
            suffix = f" Debug: {screenshot_path}" if screenshot_path else ""
            self._log(f'[Macro] "nav_unitmanager" not found within {timeout:.0f}s -- never teleported '
                       f'in-game (or the Unit Manager button isn\'t matching your setup -- if you\'re '
                       f'visibly in the match, add your own crop of it via Settings > General > '
                       f'Image Manager). Stopping.{suffix}')
        return False

    def _wait_for_teleport_result(self, hwnd, stop_event: threading.Event, timeout: float,
                                  lobby_grace: float = 0.0) -> str:
        """Poll for teleport success or Roblox's definite disconnect prompt.

        ``teleportstuck`` is only Roblox's ordinary black loading screen, not
        a distinct error state. Normal loading gets the caller's full timeout;
        only the Reconnect/Retry prompt is an immediate failure. Returns
        "ok", "disconnected", "lobby", "stopped", or "timeout"."""
        deadline = time.time() + timeout
        started_at = time.time()
        lobby_sightings = 0     # consecutive checks that found the lobby's Play button
        polls = 0
        while True:
            if polls and time.time() >= deadline:
                return "timeout"
            if stop_event.is_set():
                return "stopped"
            try:
                match = vision.find_image(hwnd, "nav_unitmanager")
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] Can't confirm teleport-in: {exc}")
                return "timeout"
            if match is not None:
                return "ok"

            for name in RECONNECT_IMAGE_NAMES:
                try:
                    reconnect_match = vision.find_image(hwnd, name)
                except vision.TemplateNotFound:
                    continue  # that particular crop hasn't been added -- try the next one
                if reconnect_match is not None:
                    return "disconnected"

            # Still on the lobby. Matchmaking was left, cancelled, or never
            # took -- either way no teleport is coming, and waiting out the
            # full timeout (5 minutes for matchmaking) achieves nothing. Play
            # only renders on the lobby, which is what makes it safe to read
            # here: during a real teleport the lobby is already gone.
            # Confirmed over consecutive checks, since the lobby is briefly
            # still drawn as a teleport begins.
            #
            # Rate-limited: this loop polls fast and can run for minutes, so a
            # full-window search every tick would be real cost for a state that
            # does not change that quickly.
            polls += 1
            if polls % LOBBY_CHECK_EVERY_N_POLLS:
                time.sleep(TELEPORT_POLL_INTERVAL)
                continue
            try:
                lobby_match = vision.find_image(hwnd, "nav_play")
            except vision.TemplateNotFound:
                lobby_match = None
            if lobby_match is None:
                lobby_sightings = 0
            else:
                lobby_sightings += 1
                if (lobby_sightings >= LOBBY_RESYNC_CONFIRMATIONS
                        and time.time() - started_at >= lobby_grace):
                    return "lobby"

            time.sleep(TELEPORT_POLL_INTERVAL)
        return "timeout"

    def _handle_disconnect(self, hwnd, stop_event: threading.Event, webhook: dict,
                             task: dict) -> None:
        """Recover after Roblox displays its definite Reconnect/Retry prompt.

        The deep link can't reconnect a client wedged on that prompt, so the
        rejoin path closes Roblox entirely (crash-equivalent end state) and
        relaunches a fresh client into the experience -- see
        _attempt_rejoin/_attempt_rejoin_locked."""
        why = "Roblox's own Reconnect/Retry prompt appeared"
        self._log(f"[Macro] Disconnected from Roblox ({why}) -- closing and relaunching Roblox.")
        screenshot_path = self._save_debug_screenshot_unconditional(hwnd, "teleport_disconnected")
        self._send_event_webhook(webhook, task, "Disconnected -- Restarting Roblox",
                                   f"{why.capitalize()}. Closing the stuck client and launching a fresh one.",
                                   0xE8935A, screenshot_path)
        self._attempt_rejoin(hwnd, stop_event)

    def _send_event_webhook(self, webhook: dict, task: dict, title: str, description: str, color: int,
                              screenshot_path: str = None, extra_fields: list = None) -> None:
        """Shared by every "worth a Discord ping" runner event that ISN'T a
        Victory/Defeat result (see _send_result_webhook for that one) --
        disconnects, a Restart Game, a Start Game click that never actually
        registered, a task finally giving up after every recovery attempt.
        Reuses the same task-result webhook config (url/enabled/mention/
        silent) rather than a second separate webhook, and attaches a
        screenshot via webhook.send_file when one's given so these read as
        "here's what was actually on screen", not just a line of text."""
        url = (webhook or {}).get("url")
        if not url or not webhook.get("enabled"):
            return
        from . import webhook as webhook_module
        fields = [{"name": "Map", "value": (task or {}).get("map") or "-", "inline": True}]
        if extra_fields:
            fields.extend(extra_fields)
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "footer": {"text": "VFERDZ MACRO | Anime Expeditions"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mention_id = (webhook or {}).get("mention_id")
        content = f"<@{mention_id}>" if mention_id else ""
        silent = bool(webhook.get("silent"))
        try:
            if screenshot_path:
                result = webhook_module.send_file(url, embed, screenshot_path, content=content, silent=silent)
            else:
                result = webhook_module.send(url, embed, content=content, silent=silent)
            if not result.get("ok"):
                self._log(f'[Macro] "{title}" webhook send failed: {result.get("reason")}')
        except Exception as exc:
            self._log(f'[Macro] "{title}" webhook send failed: {exc}')

    @staticmethod
    def _next_task_progress(task_index: int, task_count: int) -> str:
        if task_index < task_count:
            next_task = f"Task {task_index + 1}/{task_count}"
        else:
            next_task = f"Task 1/{task_count} on the next queue pass"
        return f"Auto resource checks, then {next_task}"

    def _next_challenge_progress(self) -> str:
        active = getattr(self, "_active_task_progress", None)
        next_step = "Check for another ready Challenge stage"
        if not active:
            return f"{next_step}, then Task queue"
        repeat = active.get("next_repeat")
        total = active.get("repeat_total")
        repeat_text = f" (repeat {repeat}/{total})" if repeat and total else ""
        return f"{next_step}, then Resume Task {active['index']}/{active['count']}{repeat_text}"

    def _send_progress_webhook(self, webhook: dict, task: dict, title: str, description: str,
                               color: int, extra_fields: list = None,
                               current_action: str = None, next_phase: str = None) -> None:
        """Send an optional task/challenge lifecycle notification.

        Progress is deliberately separate from the existing event alerts:
        users who only want result/failure messages keep the old behavior,
        while the opt-in setting can report queue movement as it happens.
        """
        if not (webhook or {}).get("enabled") or not (webhook or {}).get("progress"):
            return
        fields = []
        if current_action:
            fields.append({"name": "Current", "value": current_action, "inline": False})
        if next_phase:
            fields.append({"name": "Next", "value": next_phase, "inline": False})
        if extra_fields:
            fields.extend(extra_fields)
        self._send_event_webhook(
            webhook, task, title, description, color,
            extra_fields=fields or None)

    def _attempt_rejoin(self, hwnd, stop_event: threading.Event) -> bool:
        """Serialize Roblox deep-link rejoin attempts.

        Roblox/Bloxstrap owns the actual launcher process and may still be
        waiting on a single-instance handoff after this method times out. A
        non-blocking lock prevents a concurrent caller from launching another
        link, while a pending flag makes later recovery passes poll the
        existing handoff instead of launching another link.
        """
        if stop_event.is_set():
            return False
        if not self._rejoin_lock.acquire(blocking=False):
            self._log("[Macro] A Roblox rejoin is already in progress -- not launching another instance.")
            return False
        try:
            return self._attempt_rejoin_locked(hwnd, stop_event)
        finally:
            self._rejoin_lock.release()

    def _attempt_rejoin_locked(self, hwnd, stop_event: threading.Event) -> bool:
        """Closes any stuck client, launches the Roblox deep link, and waits
        for the lobby (nav_play) to come back, polling self._hwnd_getter()
        rather than trusting the original hwnd -- the deep link spawns a
        brand new process/window, and main.py's dock watchdog re-docks it
        under a NEW hwnd on its own; this is how a rejoin picks that up
        instead of continuing to poll a dead window handle. Updates
        self._current_hwnd on success. Returns whether the lobby was
        actually reached again."""
        if self._rejoin_pending:
            # The previous call already handed the link to Roblox/Bloxstrap.
            # Its launcher can remain alive after our timeout, so keep polling
            # the same handoff instead of opening a second link/instance.
            self._set_status(action="Rejoin pending -- waiting for the lobby...")
            self._log("[Macro] A Roblox rejoin is already pending -- waiting on the existing launch "
                      "instead of opening another instance.")
        else:
            # A deep-link launch invokes Roblox's OWN single-instance handling,
            # which force-closes every OTHER open Roblox window down to just
            # the newly launched one -- fine (even desired) on a single-
            # instance setup, but it was silently taking out someone's other
            # accounts/windows on a multi-instance one, which is a much worse
            # outcome than just failing this one rejoin attempt. Only attempt
            # it when there's nothing else around it could take down (the
            # window that actually disconnected doesn't count here -- it's
            # still docked/hidden at this point, not a standalone window
            # list_roblox_windows would even see).
            try:
                other_windows = wm.list_roblox_windows()
            except Exception:
                other_windows = []
            # In cutout mode (e.g. use_wgc_capture) our OWN Roblox stays a
            # top-level window, so it shows up in list_roblox_windows and this
            # guard tripped on our own game -- a rejoin would then never fire.
            # Exclude the window we're farming (list_roblox_windows returns dicts)
            # so only GENUINELY other Roblox windows (alt accounts) block a rejoin.
            # Harmless in the normal child-reparent dock, where our own window
            # isn't top-level and wouldn't be listed here anyway.
            try:
                _cur = self._hwnd_getter() if self._hwnd_getter else hwnd
            except Exception:
                _cur = hwnd
            other_windows = [w for w in other_windows
                             if (w.get("hwnd") if isinstance(w, dict) else w) not in (hwnd, _cur)]
            if other_windows:
                self._log("[Macro] Not attempting a deep-link rejoin -- other Roblox windows are open and it "
                           "would close them. Continuing without launching another instance.")
                return False

            self._set_status(action="Disconnected -- restarting Roblox...")
            # A disconnect wedges the client on its Reconnect/Retry prompt;
            # the deep link alone can't recover that -- Roblox never answers
            # it. Close the stuck client outright (same end state as a
            # crash) so the deep link below boots a FRESH Roblox instead of
            # asking a dead session to reconnect. Only genuine crashes got
            # here before, and closing the window ourselves reproduces that
            # path exactly.
            try:
                if _cur and wm.is_window(_cur):
                    wm.close_roblox_process(_cur)
                    self._log("[Macro] Closed the disconnected Roblox client -- launching a fresh one.")
                    # Let the process actually die before the deep link's
                    # single-instance handling talks to a half-gone client.
                    time.sleep(1.0)
            except Exception as exc:
                self._log(f"[Macro] Couldn't close the disconnected Roblox client: {exc}")
            # A rejoin creates a fresh game session; the previous team's visual
            # state cannot be assumed to survive it.
            self._last_applied_team_loadout = None
            try:
                _open_deep_link(REJOIN_DEEPLINK)
            except (OSError, webbrowser.Error) as exc:
                self._log(f"[Macro] Couldn't launch the rejoin link: {exc}")
                return False
            self._rejoin_pending = True
            self._log("[Macro] Rejoin link launched -- waiting for the game to load back in...")

        deadline = time.time() + REJOIN_TIMEOUT
        last_hwnd = hwnd
        while time.time() < deadline:
            if stop_event.is_set():
                return False
            time.sleep(REJOIN_POLL_INTERVAL)
            current_hwnd = self._hwnd_getter() if self._hwnd_getter else hwnd
            if not current_hwnd or not wm.is_window(current_hwnd):
                continue
            last_hwnd = current_hwnd
            try:
                match, _ = vision.find_image_any(current_hwnd, NAV_PLAY_IMAGE_NAMES)
            except vision.TemplateNotFound:
                match = None
            if match is not None:
                self._log("[Macro] Rejoined -- back on the lobby.")
                self._current_hwnd = current_hwnd
                self._rejoin_pending = False
                return True
        screenshot_path = self._save_debug_screenshot_unconditional(last_hwnd, "rejoin_timeout")
        suffix = f" Debug: {screenshot_path}" if screenshot_path else ""
        self._set_status(action="Rejoin pending -- still waiting for the lobby...")
        self._log(f"[Macro] Rejoin didn't reach the lobby within {REJOIN_TIMEOUT:.0f}s -- keeping the "
                  f"existing launch pending; no second deep link will be opened.{suffix}")
        return False

    def _click_start_and_wait_teleport(self, hwnd, stop_event: threading.Event, webhook: dict = None,
                                          task: dict = None) -> bool:
        """Solo mode's Start click used to fire once, and if it didn't
        actually register in-game (a frame too early, an overlay in the way)
        the run just sat there until the teleport wait ran out with nothing
        to show for it. Retrying the CLICK itself -- not just waiting longer
        -- is what actually recovers from that.

        The bug this fixes: after a successful click, nav_start legitimately
        disappears from screen (you're on your way to the loading/teleport
        screen) -- treating "can't find nav_start anymore" as a failed
        attempt and giving up after SOLO_START_RETRY_ATTEMPTS was punishing
        a teleport that was just SLOW, not one that never started. Only
        re-click when nav_start is still actually visible (the click really
        didn't register); once it's gone, that's success -- keep waiting for
        nav_unitmanager instead of trying to click a button that isn't there.
        """
        clicked = False
        for _ in range(1, SOLO_START_RETRY_ATTEMPTS + 1):
            if self._checkpoint(stop_event):
                return False

            # wait_for_image, not a bare one-shot find_image: right after
            # nav_select_stage is clicked, this screen is still mid-
            # transition for a moment, and a single check timed unluckily
            # can miss a button that's genuinely there a beat later. Only on
            # a LATER attempt (clicked already) is a quick poll actually
            # useful signal about whether it's really gone.
            try:
                start_match, start_name = vision.wait_for_image_any(
                    hwnd, NAV_START_IMAGE_NAMES, timeout=SOLO_START_TIMEOUT, stop_event=stop_event)
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] {exc}")
                return False
            if self._checkpoint(stop_event):
                return False
            if start_match is not None:
                debug_path = self._debug_save(hwnd, start_name, start_match)
                suffix = f" Debug: {debug_path}" if debug_path else ""
                self._log(f'[Macro] Found "{start_name}" (score {start_match["score"]:.2f}) -- clicking it.{suffix}')
                self._set_status(action="Clicking Start...")
                vision.click_match(self._mouse, hwnd, start_match)
                clicked = True
                if task and task.get("mode") == "boss_rush":
                    self._log(f"[Boss Rush] Waiting {BOSS_RUSH_TRANSITION_DELAY:.0f}s after Start.")
                    self._interruptible_sleep(BOSS_RUSH_TRANSITION_DELAY, stop_event)
            elif not clicked:
                # Never managed to click it even once, and it's already
                # gone -- this is the wrong screen entirely, not a slow
                # teleport, so there's nothing to keep waiting on.
                self._log(f'[Macro] "nav_start" not found within {SOLO_START_TIMEOUT:.0f}s -- the Start '
                           f'button never showed up (if it\'s visibly on screen, add your own crop of it '
                           f'via Settings > General > Image Manager). Stopping.')
                return False
            else:
                self._log("[Macro] Start already clicked -- still teleporting, waiting longer.")
            if self._checkpoint(stop_event):
                return False

            self._log(f'[Macro] Waiting to teleport in-game (watching for "nav_unitmanager", up to '
                       f'{SOLO_TELEPORT_PER_ATTEMPT_TIMEOUT:.0f}s)...')
            self._set_status(action='Waiting to teleport in-game ("nav_unitmanager")...')
            result = self._wait_for_teleport_result(
                hwnd, stop_event, SOLO_TELEPORT_PER_ATTEMPT_TIMEOUT)
            if result == "ok":
                self._log("[Macro] Teleported in-game.")
                return True
            if result == "disconnected":
                # Broken, not slow -- Roblox displayed its definite
                # Reconnect/Retry prompt, so rejoin instead of retrying Start.
                self._handle_disconnect(hwnd, stop_event, webhook, task)
                return False
            if stop_event.is_set():
                return False
            self._log("[Macro] Didn't teleport yet -- checking again.")

        self._log(f'[Macro] "nav_unitmanager" never matched across {SOLO_START_RETRY_ATTEMPTS} Start '
                   f'attempts -- never teleported in-game, stopping.')
        return False

    def _click_found_image(self, hwnd, name: str, timeout: float, stop_event: threading.Event = None,
                            shuffle: bool = False, threshold: float = vision.DEFAULT_THRESHOLD,
                            region: tuple = None, color: bool = False) -> dict:
        """Shared wait-for-it-then-click for a plain nav button (nav_settings,
        nav_search, ...) -- no per-button quirks like Story/Play have, so one
        helper covers all of them instead of a bespoke method each.

        shuffle=True hovers into the button with real relative moves before
        clicking (see vision.click_match) -- for buttons that need genuine
        hover-in movement to register the click (the lobby Event button).

        Region allows us to isolate part of the 1152x756 screen resolution
        This was made for the Tower game mode where each element is similar
        to one another.

        Returns the match dict (truthy) on success or None (falsy) on
        failure -- existing `if not self._click_found_image(...)` call sites
        work unchanged either way, but callers that need the click position
        afterward (see _start_game_or_reset_via_settings saving nav_search's
        spot to reuse) can read it straight off the returned match instead of
        re-searching for the same button a second time.
        """
        try:
            if color:
                # Color matching distinguishes stateful UI art, but grayscale
                # remains a useful fallback when capture compression or a
                # graphics setting changes the colors of a saved crop.
                color_timeout = timeout / 2
                match = vision.wait_for_color_image(
                    hwnd, name, region=region, timeout=color_timeout,
                    threshold=threshold, stop_event=stop_event)
                if match is None and (stop_event is None or not stop_event.is_set()):
                    match = vision.wait_for_image(
                        hwnd, name, region=region, timeout=timeout - color_timeout,
                        threshold=threshold, stop_event=stop_event)
            else:
                match = vision.wait_for_image(
                    hwnd, name, region=region, timeout=timeout,
                    threshold=threshold, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return None
        if match is None:
            if stop_event is None or not stop_event.is_set():
                self._log(f'[Macro] "{name}" not found within {timeout:.0f}s -- stopping.')
            return None
        debug_path = self._debug_save(hwnd, name, match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f'[Macro] Found "{name}" (score {match["score"]:.2f}) -- clicking it.{suffix}')
        vision.click_match(self._mouse, hwnd, match, shuffle=shuffle)
        return match

    def _click_and_verify_gone(self, hwnd, stop_event: threading.Event, name: str, timeout: float,
                                 retry_attempts: int = 3, verify_settle: float = 1.0,
                                 success_name: str = None, require_gone: bool = False) -> bool:
        """Like _click_found_image, but re-checks the button actually
        disappeared afterward and re-clicks (with a focus reassert) if it's
        still there, up to retry_attempts times -- same "click found it but
        Roblox never got it" flakiness the Play/Start Game clicks needed
        this same treatment for. Used for Leave Stage/Repeat Stage: a
        dropped click here leaves the whole run just sitting on the result
        screen indefinitely, since nothing else would ever notice or retry
        on its own. ``success_name`` handles transitions where the original
        image remains visible behind the next modal: Leave Stage is still
        visible when Return to Lobby appears, so seeing ``return`` is also
        proof that the click registered.

        By default, returns whether the button was found at all -- callers
        can opt into requiring it to disappear after the final click."""
        match = None
        for attempt in range(1, retry_attempts + 1):
            try:
                match = vision.wait_for_image(hwnd, name, timeout=timeout, stop_event=stop_event)
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] {exc}")
                return False
            if match is None:
                if stop_event is None or not stop_event.is_set():
                    self._log(f'[Macro] "{name}" not found within {timeout:.0f}s -- stopping.')
                return False

            debug_path = self._debug_save(hwnd, name, match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f'[Macro] Found "{name}" (score {match["score"]:.2f}) -- clicking it '
                       f'(attempt {attempt}/{retry_attempts}).{suffix}')
            if not wm.activate_window(hwnd):
                self._log(f"[Macro] Couldn't confirm focus before clicking \"{name}\" -- click may not register.")
            vision.click_match(self._mouse, hwnd, match)
            time.sleep(verify_settle)
            if self._checkpoint(stop_event):
                return True

            if success_name is not None:
                try:
                    transitioned = vision.find_image(hwnd, success_name)
                except vision.TemplateNotFound:
                    transitioned = None
                if transitioned is not None:
                    return True

            try:
                still_there = vision.find_image(hwnd, name)
            except vision.TemplateNotFound:
                still_there = None
            if still_there is None:
                return True
            if attempt == retry_attempts:
                if require_gone:
                    self._log(f'[Macro] "{name}" still showing after {retry_attempts} clicks.')
                    return False
                self._log(f'[Macro] "{name}" still showing after {retry_attempts} clicks -- continuing anyway.')
        return True

    def _wait_for_image_gone(self, hwnd, names, timeout: float, stop_event: threading.Event = None) -> bool:
        """Polls until NONE of `names` are found anymore, or timeout runs
        out. No clicking -- purely a "has this actually disappeared yet"
        wait, for confirming a modal/banner is really gone rather than just
        assuming it is because some OTHER click nearby reported success
        (see _handle_match_result's Repeat Stage path, where the button
        image itself disappearing turned out not to mean the whole result
        panel had). Best-effort: a missing template is silently treated as
        already-gone (same as any optional check in this file), and running
        out the timeout just returns False rather than raising -- callers
        treat this as a settle wait, not a hard requirement."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if stop_event is not None and stop_event.is_set():
                return False
            still_showing = False
            for name in names:
                try:
                    match = vision.find_image(hwnd, name)
                except vision.TemplateNotFound:
                    continue
                if match is not None:
                    still_showing = True
                    break
            if not still_showing:
                return True
            time.sleep(0.3)
        self._log(f'[Macro] {"/".join(names)} still showing after {timeout:.0f}s -- continuing anyway.')
        return False

    def _click_start_game_2_if_found(self, hwnd) -> bool:
        # nav_start_game_confirm (was "nav_start_game_2" -- renamed because
        # the _2 suffix made it look like a mere visual variant of
        # nav_start_game, which it is NOT): a second Start Game/confirm
        # button that can show up alongside the warning itself (e.g. a
        # "Start Anyway" prompt) -- clicking it skips the wait in
        # _wait_out_start_game_warning entirely instead of sitting through
        # the full timeout for a warning that's actually already
        # dismissable right now.
        try:
            skip_match = vision.find_image(hwnd, "nav_start_game_confirm")
        except vision.TemplateNotFound:
            return False
        if skip_match is None:
            return False
        self._log(f"[Macro] Found nav_start_game_confirm (score {skip_match['score']:.2f}) -- "
                   f"clicking it to skip the warning wait.")
        vision.click_match(self._mouse, hwnd, skip_match)
        return True

    def _find_start_game_button(self, hwnd, stop_event: threading.Event = None, timeout: float = 0):
        """Tries nav_start_game (whose folder holds every visual variant of
        the ready-up button seen in practice -- the old separately-named
        _3/_4 crops live in there now, all tried automatically per search),
        then nav_start_game_confirm (a DIFFERENT button -- the "Start
        Anyway"-style second confirm, see _click_start_game_2_if_found) --
        so the actual "start the round" click (see _play_one_match) isn't
        dependent on just one image matching. Returns (name, match) for
        whichever was found first, or (None, None) if none of them were --
        missing/not-yet-added templates are skipped silently, same as any
        other optional template.

        timeout=0 (the default) is a single instant pass -- used right
        after a click to check it's gone, where waiting around would just
        slow the retry loop down. Pass a real timeout for the FIRST check
        (right as Pre Start hands off), since the button can still be
        animating in at that exact moment and a one-shot check there was
        landing before it existed at all, especially on Expedition where
        Pre Start's place_unit clicks run right up until this point."""
        deadline = time.time() + max(0.0, timeout)
        while True:
            for name in ("nav_start_game", "nav_start_game_confirm"):
                try:
                    match = vision.find_image(hwnd, name)
                except vision.TemplateNotFound:
                    continue
                if match is not None:
                    return name, match
            if time.time() >= deadline or (stop_event is not None and stop_event.is_set()):
                return None, None
            time.sleep(0.3)

    def _wait_out_start_game_warning(self, hwnd, stop_event: threading.Event) -> None:
        """Best-effort, like nav_disband: a warning popup (e.g. an
        incomplete-setup warning) can sit in front of Start Game right
        after Pre Start finishes -- if one's up, wait for it to clear (and
        Start Game to actually show up) instead of immediately treating a
        missing nav_start_game as "already started" and moving on. A
        missing warning.png just skips this check entirely, same as any
        other optional template."""
        try:
            warning_match = vision.find_image(hwnd, "warning")
        except vision.TemplateNotFound:
            return
        if warning_match is None:
            return

        self._log(f"[Macro] Found a warning (score {warning_match['score']:.2f}) -- checking for a way past it.")
        if self._click_start_game_2_if_found(hwnd):
            return

        self._log(f"[Macro] Waiting up to {WARNING_WAIT_TIMEOUT:.0f}s for it to clear.")
        self._set_status(action="Waiting for warning to clear...")
        deadline = time.time() + WARNING_WAIT_TIMEOUT
        while time.time() < deadline:
            if self._checkpoint(stop_event):
                return
            if self._click_start_game_2_if_found(hwnd):
                return
            try:
                warning_gone = vision.find_image(hwnd, "warning") is None
            except vision.TemplateNotFound:
                warning_gone = True
            try:
                start_visible = vision.find_image(hwnd, "nav_start_game") is not None
            except vision.TemplateNotFound:
                start_visible = False
            if warning_gone and start_visible:
                self._log("[Macro] Warning cleared -- Start Game is up.")
                return
            time.sleep(WARNING_POLL_INTERVAL)
        self._log(f'[Macro] "warning" still showing (or "nav_start_game" still not found) after '
                   f'{WARNING_WAIT_TIMEOUT:.0f}s -- continuing anyway.')

    def _open_settings_search(self, hwnd, stop_event: threading.Event):
        """Opens Settings and clicks its search box. Returns the search
        box's screen position (reusable for as many searches as needed --
        the box itself never moves once found, so later callers don't need
        a second image search against a panel that's since reflowed around
        whatever the first search changed), or None if Settings/the search
        box couldn't be found at all."""
        if not self._click_found_image(hwnd, "nav_settings", NAV_CLICK_TIMEOUT, stop_event):
            return None
        if self._checkpoint(stop_event):
            return None
        # The Settings panel opens with a scale/slide-in animation -- without
        # this, nav_search can get matched (even at a perfect 1.00 score)
        # against a mid-animation frame, whose search box isn't at its final
        # settled position yet. The click then lands wherever that transient
        # frame put it instead of where the box actually ends up, missing it
        # entirely. SETTLE_DELAY is comfortably longer than the animation.
        time.sleep(SETTLE_DELAY)
        search_match = self._click_found_image(hwnd, "nav_search", NAV_CLICK_TIMEOUT, stop_event)
        if not search_match:
            return None
        if self._checkpoint(stop_event):
            return None
        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        return (left + search_match["cx"], top + search_match["cy"])

    def _search_and_set_toggle(self, hwnd, stop_event: threading.Event, search_box_pos, setting_name: str,
                                 desired_on: bool) -> bool:
        """Types setting_name into an already-open Settings search box
        (see _open_settings_search) and clicks its toggle if it isn't
        already in the desired on/off state. Shared by the Auto Vote Start
        handling below and any Setting block of "toggle" kind (see
        _run_setting_block) -- same search-and-toggle mechanic either way,
        just a different setting name/desired state. Returns whether the
        setting's toggle was found at all (not whether a click happened --
        already being in the right state is also success)."""
        self._set_status(action=f'Searching settings for "{setting_name}"...')
        self._log(f'[Macro] Typing "{setting_name}"...')
        self._mouse.click(*search_box_pos)
        time.sleep(0.2)  # let the search field actually take focus before typing
        self._keyboard.combo(keys.VK_CONTROL, ord("A"))  # select any existing search text...
        self._keyboard.tap(keys.VK_DELETE)                # ...and clear it before typing this search
        self._keyboard.type_text(setting_name)
        time.sleep(SETTLE_DELAY)  # let the filtered settings list render before reading its toggle
        if self._checkpoint(stop_event):
            return False

        try:
            on_match = vision.find_image(hwnd, "toggle_true")
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        is_on = on_match is not None

        if is_on == desired_on:
            self._log(f'[Macro] "{setting_name}" already {"on" if desired_on else "off"} -- nothing to click.')
            return True

        # Not in the desired state -- toggle_true/toggle_false share the
        # same click target either way (clicking the switch flips it), so
        # whichever of the two actually matched is what gets clicked.
        toggle_match = on_match
        if toggle_match is None:
            try:
                toggle_match = vision.find_image(hwnd, "toggle_false")
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] {exc}")
                return False
        if toggle_match is None:
            self._log(f'[Macro] "{setting_name}" not found in Settings -- skipping.')
            return False
        debug_path = self._debug_save(hwnd, "toggle_true" if is_on else "toggle_false", toggle_match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f'[Macro] "{setting_name}" is {"on" if is_on else "off"} -- '
                   f'turning it {"off" if is_on else "on"}.{suffix}')
        vision.click_match(self._mouse, hwnd, toggle_match)
        return True

    def _close_settings_if_open(self, hwnd, stop_event: threading.Event) -> None:
        """One-shot cleanup shared by anything that opened Settings via
        _open_settings_search and is done with it -- a look, not a wait,
        since not finding it open is success too (some flows, like a
        Restart Game confirm, can already close Settings on their own)."""
        time.sleep(SETTLE_DELAY)
        if self._checkpoint(stop_event):
            return
        try:
            settings_match = vision.find_image(hwnd, "nav_settings_on")
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return
        if settings_match is None:
            return
        debug_path = self._debug_save(hwnd, "nav_settings_on", settings_match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f"[Macro] Closing Settings.{suffix}")
        vision.click_match(self._mouse, hwnd, settings_match)

    def _start_game_or_reset_via_settings(self, hwnd, stop_event: threading.Event, play_mode: str = "solo") -> bool:
        # Party leadership and Auto Vote Start are both matchmaking-only
        # concepts -- Solo mode has no party at all, so there's no leader
        # to check for and nothing to reset via Settings. This used to run
        # unconditionally for solo too, which is exactly what produced the
        # confusing "No Start Game button (not the party leader)" log on
        # every solo run: there was never a party to be leader of, so the
        # check always "failed" and fell through to the Auto Vote Start
        # search, a setting that's entirely irrelevant to a solo run.
        if play_mode != "matchmaking":
            return True

        # Start Game only exists for the party leader -- a quick presence
        # check, not a long wait (see START_GAME_CHECK_TIMEOUT), decides
        # which of the two very different paths to take. Either way, this is
        # a LEADERSHIP check, not the actual "begin the round" click: Start
        # Game must not be pressed until Pre Start (camera, walk, unit
        # placement) has actually run -- pressing it here, this early, was
        # starting the round before any of that had a chance to happen.
        # The real click lives at the end of _run, once, after Pre Start
        # finishes, for leader and non-leader alike.
        self._set_status(action="Checking party leadership...")
        try:
            match = vision.wait_for_image(hwnd, "nav_start_game", timeout=START_GAME_CHECK_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        if stop_event.is_set():
            return False
        if match is not None:
            self._log(f"[Macro] Found Start Game (score {match['score']:.2f}) -- you're the party leader. "
                       f"Running Pre Start before pressing it.")
            return True

        # No Start Game button just means Auto Vote Start is on -- most
        # people run with it on deliberately (it's a legitimate feature,
        # not a misconfiguration), so this no longer fights it by disabling
        # it and restarting the game. The vote-started round teleports in
        # on its own; _wait_teleport_in downstream is what actually confirms
        # that happened, this just needs to not treat its absence as a
        # failure.
        self._log('[Macro] No Start Game button found -- Auto Vote Start is likely on, letting it '
                   "auto-start the round instead of disabling it.")
        return True

    def _optional_cxy(self, prefix: str):
        """Return an optional coordinate override, or ``None`` for Auto."""
        x = self._coords.get(f"{prefix}_x")
        y = self._coords.get(f"{prefix}_y")
        if x in (None, "") or y in (None, ""):
            return None
        try:
            return int(x), int(y)
        except (TypeError, ValueError):
            return None

    def _cxy(self, prefix: str) -> tuple:
        """(x, y) for a Macro Coordinates point -- self._coords holds the
        user's saved overrides (or DEFAULT_COORDS before a run starts), so
        every click site that reads through here is re-tunable from
        Settings > Debug without a code change."""
        return int(self._coords[f"{prefix}_x"]), int(self._coords[f"{prefix}_y"])

    def _select_difficulty(self, hwnd, difficulty: str, coords: dict) -> None:
        # Fixed spot on the stage-detail panel, same as the stage rows --
        # no image search needed, just like Story's click was.
        key_prefix = "difficulty_hard" if difficulty == "Hard" else "difficulty_normal"
        x, y = coords[f"{key_prefix}_x"], coords[f"{key_prefix}_y"]
        self._log(f'[Macro] Clicking difficulty "{difficulty}" at ({x}, {y}).')
        self._set_status(action=f'Clicking difficulty "{difficulty}"...')
        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        self._mouse.click(left + x, top + y)


    def _click_enter_matchmaking(self, hwnd, stop_event: threading.Event, coords: dict,
                                 mode: str = None, story_event: bool = False) -> bool:
        # Expedition's and Challenge's matchmaking buttons are each their
        # own image (exp_enter_matchmaking / chal_enter) at an uncalibrated
        # position -- no matchmaking_region_* exists for either, so they're
        # searched full-window instead of the Story/Raid region restriction.
        image_name = {"expedition": "exp_enter_matchmaking", "challenge": "chal_enter"}.get(mode, "enter_matchmaking")
        # Event reuses Story/Raid's shared "enter_matchmaking" image, but its
        # button isn't at the calibrated Story/Raid position (matchmaking_
        # region_*) -- so it's searched full-window, same as Expedition/
        # Challenge, rather than boxed to a region it may not land in.
        region = None if mode in ("expedition", "challenge", "event", "summer", "boss_rush") or story_event else (
            coords["matchmaking_region_x"], coords["matchmaking_region_y"],
            coords["matchmaking_region_w"], coords["matchmaking_region_h"],
        )
        self._log(f'[Macro] Waiting for Enter Matchmaking (searching "{image_name}", up to '
                   f'{MATCHMAKING_WAIT_TIMEOUT:.0f}s)...')
        self._set_status(action=f'Waiting for Enter Matchmaking ("{image_name}")...')
        try:
            match = vision.wait_for_image(
                hwnd, image_name, region=region,
                timeout=MATCHMAKING_WAIT_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't find Enter Matchmaking: {exc}")
            return False
        if match is None:
            if not stop_event.is_set():
                self._log(f'[Macro] "{image_name}" not found within {MATCHMAKING_WAIT_TIMEOUT:.0f}s -- the '
                           f'Enter Matchmaking button never showed up (if it\'s visibly on screen, add your '
                           f'own crop of it via Settings > General > Image Manager). Stopping.')
            return False
        debug_path = self._debug_save(hwnd, image_name, match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f"[Macro] Found Enter Matchmaking (score {match['score']:.2f}) -- clicking it.{suffix}")
        vision.click_match(self._mouse, hwnd, match)
        return True

    def _select_stage(self, hwnd, stop_event: threading.Event, stage: str, mode: str,
                      story_event: str = "Normal") -> bool:
        # Raid's screen is the same nav_select_stage screen as Story's, just
        # with 3 Act rows spaced differently instead of the 7 stage rows
        # (see ACT_ORDER/ACT_CLICK_BASE/ACT_ROW_HEIGHT).
        order, base, row_height, label = (
            (ACT_ORDER, self._cxy("act_row"), int(self._coords["act_row_height"]), "Act") if mode == "raid"
            else (STAGE_ORDER, self._cxy("stage_row"), int(self._coords["stage_row_height"]), "stage"))
        event_stage = mode == "story" and story_event not in (None, "", "Normal")
        if stage not in order and not event_stage:
            self._log(f'[Macro] Unknown {label} "{stage}" -- expected one of {order}.')
            return False

        self._log("[Macro] Waiting for the stage select screen...")
        self._set_status(action="Waiting for stage screen...")
        try:
            match = vision.wait_for_image(
                hwnd, "nav_select_stage", timeout=STAGE_SCREEN_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't confirm the stage screen opened: {exc}")
            return False
        if match is None:
            if not stop_event.is_set():
                self._log(f'[Macro] "nav_select_stage" not found within {STAGE_SCREEN_TIMEOUT:.0f}s -- the '
                           f'stage select screen never opened (if it\'s visibly open, add your own crop of '
                           f'the Select Stage button via Settings > General > Image Manager). Stopping.')
            return False

        # nav_select_stage confirms the CONFIRM BUTTON is on screen, not
        # that the stage/Act row list above it has finished rendering that
        # map's own rows yet -- clicking a computed row position immediately
        # used to be able to catch the list still mid-transition (e.g. a
        # moment of the PREVIOUS map's rows still visible/animating out),
        # landing on the wrong stage entirely (reported: Mastery selected,
        # but a regular numbered stage started instead).
        time.sleep(SETTLE_DELAY)

        visual_name = (
            STORY_EVENT_STAGE_IMAGES.get(story_event)
            if event_stage
            else STORY_STAGE_VISUAL_IMAGES.get(stage) if mode == "story" else None
        )
        if visual_name:
            return self._select_story_stage_visually(
                hwnd, stop_event, story_event if event_stage else stage, visual_name,
                verify_selection=not event_stage)

        idx = order.index(stage)
        x = base[0]
        y = base[1] + idx * row_height
        self._log(f'[Macro] Stage screen open -- double-clicking {label} "{stage}" at ({x}, {y}).')
        self._set_status(action=f'Clicking {label} "{stage}"...')
        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        # A single click was sometimes not registering as the row's own
        # "selected" state (confirmed from real reports on both Story and
        # Raid) -- a double-click reliably does. The stage-detail panel
        # (Story's difficulty toggle, or straight to the Select Stage
        # confirm button for Raid/Infinite/Mastery, which have no
        # difficulty toggle at all) then animates in, same as the map click
        # before it -- settled here UNCONDITIONALLY, not just on the
        # difficulty-click path, since Raid/locked stages used to go
        # straight from this click into Select Stage with no wait at all.
        self._mouse.double_click(left + x, top + y)
        time.sleep(DIFFICULTY_CLICK_DELAY)
        return True

    @staticmethod
    def _stage_row_looks_selected(image) -> bool:
        """Whether a small stage-row sample has the selected blue fill."""
        if image is None or image.size == 0:
            return False
        # BGR. Selected rows are a saturated cyan/blue gradient; unselected
        # rows are flat neutral gray. The glyph itself occupies little of
        # this 49x49 sample, so require a substantial blue fraction instead
        # of trusting one colored edge/glow pixel.
        blue = image[:, :, 0].astype("int16")
        green = image[:, :, 1].astype("int16")
        red = image[:, :, 2].astype("int16")
        selected = (
            (blue > 100)
            & (green > 55)
            & (blue > red + 45)
            & (green > red + 25)
        )
        return float(selected.mean()) >= STORY_STAGE_SELECTED_BLUE_MIN_FRACTION

    def _story_stage_match_is_selected(self, hwnd, match: dict) -> bool:
        half = STORY_STAGE_SELECTED_SAMPLE_HALF_SIZE
        region = (
            int(match["cx"]) - half,
            int(match["cy"]) - half,
            half * 2 + 1,
            half * 2 + 1,
        )
        image = vision.capture_window_region_bgr(hwnd, region)
        return self._stage_row_looks_selected(image)

    def _select_story_stage_visually(
            self, hwnd, stop_event: threading.Event, stage: str, image_name: str,
            verify_selection: bool = True) -> bool:
        """Find a named Story row visually and prove its selected state."""
        try:
            match = vision.wait_for_image(
                hwnd, image_name, region=STORY_STAGE_SEARCH_REGION,
                threshold=STORY_STAGE_MATCH_THRESHOLD,
                timeout=STAGE_SCREEN_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't visually locate Story stage {stage}: {exc}")
            return False
        if match is None:
            if not stop_event.is_set():
                self._log(
                    f'[Macro] "{image_name}" did not match inside the stage list -- '
                    f'cannot safely select {stage}.'
                )
                self._save_debug_screenshot_unconditional(
                    hwnd, f"stage_{stage.lower()}_not_found")
            return False

        self._log(
            f'[Macro] Found stage "{stage}" visually (score {match["score"]:.2f}) '
            f'at ({match["cx"]}, {match["cy"]}).'
        )
        if verify_selection and self._story_stage_match_is_selected(hwnd, match):
            self._log(f'[Macro] Stage "{stage}" is already selected.')
            time.sleep(DIFFICULTY_CLICK_DELAY)
            return True

        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        for attempt in range(1, STORY_STAGE_CLICK_ATTEMPTS + 1):
            if self._checkpoint(stop_event):
                return False
            self._set_status(action=f'Clicking stage "{stage}"...')
            if not wm.activate_window(hwnd):
                self._log(
                    f'[Macro] Could not confirm focus before selecting stage "{stage}" '
                    "-- click may not register."
                )
            self._log(
                f'[Macro] Double-clicking stage "{stage}" at '
                f'({match["cx"]}, {match["cy"]}) '
                f'(attempt {attempt}/{STORY_STAGE_CLICK_ATTEMPTS}).'
            )
            self._mouse.double_click(left + match["cx"], top + match["cy"])

            if not verify_selection:
                # Event rows use a different selected-state treatment from the
                # standard blue Story rows. The shared Select Stage button is
                # the authoritative confirmation for these special stages.
                time.sleep(DIFFICULTY_CLICK_DELAY)
                return True

            deadline = time.time() + STORY_STAGE_SELECTED_VERIFY_TIMEOUT
            while time.time() < deadline:
                if self._checkpoint(stop_event):
                    return False
                if self._story_stage_match_is_selected(hwnd, match):
                    self._log(
                        f'[Macro] Confirmed stage "{stage}" is selected.'
                    )
                    time.sleep(DIFFICULTY_CLICK_DELAY)
                    return True
                time.sleep(STORY_STAGE_SELECTED_VERIFY_POLL)
            if attempt < STORY_STAGE_CLICK_ATTEMPTS:
                self._log(
                    f'[Macro] Stage "{stage}" did not turn selected -- retrying its row.'
                )

        self._log(
            f'[Macro] Stage "{stage}" never showed its selected state after '
            f'{STORY_STAGE_CLICK_ATTEMPTS} attempts -- stopping before the wrong stage starts.'
        )
        self._save_debug_screenshot_unconditional(
            hwnd, f"stage_{stage.lower()}_selection_failed")
        return False

    def _ensure_lobby(self, hwnd, stop_event: threading.Event) -> bool:
        # "On the lobby" is inferred from the Play button actually being
        # visible in its known Nav spot -- it only renders there outside of
        # a match, so finding it there IS the lobby check, not a separate
        # step before it.
        self._log("[Macro] Checking you're on the lobby...")
        self._set_status(action="Checking lobby...")
        try:
            match, _ = vision.wait_for_image_any(
                hwnd, NAV_PLAY_IMAGE_NAMES,
                timeout=LOBBY_CHECK_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't check the lobby: {exc}")
            return False
        if match is not None:
            return True
        if stop_event.is_set():
            return False
        # No Play button after a full LOBBY_CHECK_TIMEOUT wait looks exactly
        # like a silent disconnect that never even triggered Roblox's own
        # Reconnect/Retry prompt (see _handle_disconnect) -- rather than
        # just give up here, try the same deep-link rejoin a detected
        # disconnect already uses instead of stopping the whole run over it.
        # (_attempt_rejoin itself skips this on a multi-instance setup --
        # see its own comment.)
        self._log(f'[Macro] "nav_play" not found within {LOBBY_CHECK_TIMEOUT:.0f}s -- not on the lobby '
                   f'(likely a silent disconnect), attempting a rejoin via deep link.')
        return self._attempt_rejoin(hwnd, stop_event)

    def _dismiss_lobby_overlay(self, hwnd) -> bool:
        """Close a modal covering the lobby, e.g. the Update Log after a game
        update or a fresh login.

        Play renders behind such a modal and still matches, so the click is
        found and then lands on the overlay -- seen as Play being re-clicked
        until PLAY_CLICK_RETRY_ATTEMPTS runs out while patch notes sit on
        screen. Optional, like the party overlay: no image, no check.

        Returns True if something was closed.
        """
        try:
            match, name = vision.find_image_any(hwnd, LOBBY_OVERLAY_CLOSE_IMAGE_NAMES)
        except vision.TemplateNotFound:
            return False
        if match is None:
            return False
        self._log(f'[Macro] A lobby overlay ("{name}") is covering Play -- closing it first.')
        vision.click_match(self._mouse, hwnd, match)
        time.sleep(GAMEMODE_OVERLAY_CHECK_DELAY)
        return True

    def _click_play(self, hwnd, stop_event: threading.Event) -> bool:
        self._set_status(action="Clicking Play...")
        try:
            match, name = vision.find_image_any(hwnd, NAV_PLAY_IMAGE_NAMES)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        if match is None:
            self._log('[Macro] "nav_play" vanished before it could be clicked -- stopping.')
            return False
        debug_path = self._debug_save(hwnd, name, match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f"[Macro] Found Play (score {match['score']:.2f}) -- clicking it.{suffix}")
        # Reasserted right here, not just once at the top of the run --
        # SendInput clicks go to whatever window has REAL OS focus, and
        # this is the very first live click after a screen transition
        # (lobby just loaded), the exact moment focus is most likely to
        # not have actually settled yet. Reported by multiple users as
        # "it finds Play correctly, the click just doesn't register."
        if not wm.activate_window(hwnd):
            self._log("[Macro] Couldn't confirm focus before clicking Play -- click may not register.")
        time.sleep(0.1)
        vision.click_match(self._mouse, hwnd, match)
        # The gamemode screen replaces the lobby immediately after this
        # click. Leaving the pointer on Play can put it over a party/invite
        # control in the new layout and open an unrelated overlay. Move to
        # the existing user-calibratable empty corner before the transition.
        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        park_x, park_y = self._cxy("unit_info_reset")
        self._mouse.move_to(left + park_x, top + park_y)
        return True

    def _reach_map_selected(self, hwnd, stop_event: threading.Event, map_name: str, mode: str,
                              scroll_power: int, scroll_nudges: int, story_event: str = "Normal") -> bool:
        """Lobby -> Play -> Story/Raid -> map search, as one restartable unit --
        called in a loop by _run (see MAP_SELECT_RETRY_ATTEMPTS). Each call
        re-checks from scratch (including the "already on the gamemode
        menu" shortcut) rather than assuming any state left over from a
        previous failed attempt, since _spam_back_until_gone may have just
        backed all the way out to the lobby.
        """
        # If the gamemode menu (Story/Raid/...) is ALREADY open -- e.g. a
        # previous attempt got this far and backed out only partway, or you
        # opened it by hand -- nav_back is already on screen, so checking
        # for the lobby and clicking Play would be pointless (Play doesn't
        # even exist there) and would just burn LOBBY_CHECK_TIMEOUT waiting
        # for something that was never going to appear. One quick, non-
        # waiting check skips both steps straight to clicking Story.
        try:
            already_open = vision.find_image(hwnd, "nav_back") is not None
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        if already_open:
            self._log("[Macro] Already on the gamemode menu -- skipping the lobby and Play.")
        else:
            if not self._ensure_lobby(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False
            if not self._click_play(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False

        if not self._click_gamemode(hwnd, stop_event, mode, wait_for_menu=not already_open):
            return False
        if self._checkpoint(stop_event):
            return False

        self._set_status(action=f'Selecting map "{map_name}"...')

        # Expedition doesn't use Story's scrolling map carousel -- it's a
        # small fixed set of map cards, each found by its own reference
        # image (or, for School Grounds, not found/clicked at all -- see
        # EXPEDITION_MAP_IMAGES).
        if mode == "expedition":
            if self._select_expedition_map(hwnd, stop_event, map_name):
                return True
            self._spam_back_until_gone(hwnd, stop_event)
            return False

        log_and_status = lambda msg: (self._log(msg), self._set_status(action=msg.split("] ", 1)[-1]))
        kwargs = {"debug_screenshots": self._debug_screenshots}
        if scroll_power is not None:
            kwargs["scroll_power"] = scroll_power
        if scroll_nudges is not None:
            kwargs["scroll_nudges"] = scroll_nudges
        select_map = (
            stage_select.find_and_click_story_event
            if mode == "story" and story_event and story_event != "Normal"
            else stage_select.find_and_click_map
        )
        target = story_event if select_map is stage_select.find_and_click_story_event else map_name
        selected = select_map(self._mouse, hwnd, target, log_and_status, stop_event, **kwargs)
        if selected:
            if mode == "story" and story_event and story_event != "Normal" and isinstance(selected, str):
                self._current_task["map"] = selected
                global_settings = self._get_global_story_settings() or {}
                macro = ((global_settings.get("maps") or {}).get(selected) or {}).get("macro") or ""
                if macro:
                    self._current_task["macro"] = macro
                    self._log(f'[Macro] Using Global Story Map Setup macro "{macro}" for "{selected}".')
                else:
                    self._log(f'[Macro] No Global Story Map Setup macro is assigned for "{selected}".')
            return True

        self._spam_back_until_gone(hwnd, stop_event)
        return False

    def _reach_boss_rush_selected(self, hwnd, stop_event: threading.Event) -> bool:
        """Lobby -> Play -> Boss Rush -> its single current map, District 7."""
        try:
            already_selected = vision.find_image(hwnd, "nav_select_stage") is not None
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        if already_selected:
            self._log("[Macro] District 7 is already selected.")
            return not self._checkpoint(stop_event)

        try:
            already_open = vision.find_image(hwnd, "nav_back") is not None
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        if already_open:
            self._log("[Macro] Already on the gamemode menu -- skipping the lobby and Play.")
        else:
            if not self._ensure_lobby(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False
            if not self._click_play(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False

        if not self._click_gamemode(hwnd, stop_event, "boss_rush", wait_for_menu=not already_open):
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        self._set_status(action="Selecting District 7...")
        try:
            map_match = vision.wait_for_image(
                hwnd, "District 7", timeout=STAGE_SCREEN_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't confirm the Boss Rush map: {exc}")
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        if map_match is None:
            if not stop_event.is_set():
                self._log('[Macro] "District 7" did not appear after opening Boss Rush.')
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        debug_path = self._debug_save(hwnd, "District 7", map_match)
        suffix = f" Debug: {debug_path}" if debug_path else ""
        self._log(f'[Macro] Found District 7 (score {map_match["score"]:.2f}) -- selecting it.{suffix}')
        vision.click_match(self._mouse, hwnd, map_match)
        self._log(f"[Boss Rush] Waiting {BOSS_RUSH_TRANSITION_DELAY:.0f}s for District 7 to open.")
        self._interruptible_sleep(BOSS_RUSH_TRANSITION_DELAY, stop_event)
        if self._checkpoint(stop_event):
            return False
        try:
            stage_match = vision.wait_for_image(
                hwnd, "nav_select_stage", timeout=STAGE_SCREEN_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't confirm the District 7 selection: {exc}")
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        if stage_match is None:
            if not stop_event.is_set():
                self._log('[Macro] "Select Stage" did not appear after selecting District 7.')
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        return not self._checkpoint(stop_event)

    def _reach_summer_selection(self, hwnd, stop_event: threading.Event, task: dict,
                                scroll_power: int = None, scroll_nudges: int = None) -> bool:
        """Lobby -> Events -> Tidal Siege -> Event Gamemode -> selection."""
        summer_mode = (
            task.get("event_gamemode") if task.get("mode") == "event"
            else task.get("summer_mode")
        ) or "Event Mode"
        if summer_mode not in SUMMER_MODE_IMAGES:
            self._log(f'[Macro] Unknown Summer mode "{summer_mode}".')
            return False
        if not self._ensure_lobby(hwnd, stop_event):
            return False
        if self._checkpoint(stop_event):
            return False

        for image_name, label in (
            ("nav_event", "Events"),
            (SUMMER_TIDAL_SIEGE_IMAGE, "Tidal Siege"),
            (SUMMER_EVENT_GAMEMODE_IMAGE, "Event Gamemode"),
        ):
            self._set_status(action=f"Clicking {label}...")
            if self._click_found_image(hwnd, image_name, SUMMER_SCREEN_TIMEOUT, stop_event) is None:
                self._spam_back_until_gone(hwnd, stop_event)
                return False
            if self._checkpoint(stop_event):
                return False
            time.sleep(SETTLE_DELAY)

        mode_image = SUMMER_MODE_IMAGES[summer_mode][0]
        self._set_status(action=f"Clicking {summer_mode}...")
        if self._click_found_image(hwnd, mode_image, SUMMER_SCREEN_TIMEOUT, stop_event) is None:
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        if self._checkpoint(stop_event):
            return False
        time.sleep(SETTLE_DELAY)

        if summer_mode == "Portal Mode":
            portal = task.get("map") or "Summer Portal"
            tier = str(task.get("portal_tier") or "1")
            portal_image = SUMMER_PORTAL_IMAGES.get((portal, tier), ("",))[0]
            if tier == "Secret":
                portal_label = (
                    "Sovereign's Portal" if portal == "Summer Portal"
                    else "Lightning God's Portal"
                )
            else:
                portal_label = f"{portal} Tier {tier}"
            for image_name, label in (
                (portal_image, portal_label),
                (SUMMER_ACTIVATE_PORTAL_IMAGE, "Activate Portal"),
            ):
                if not image_name:
                    self._log(f'[Macro] Unknown Summer portal "{portal}" tier "{tier}".')
                    self._spam_back_until_gone(hwnd, stop_event)
                    return False
                self._set_status(action=f"Clicking {label}...")
                click = (
                    self._click_found_color_image(hwnd, image_name, SUMMER_SCREEN_TIMEOUT, stop_event)
                    if image_name == portal_image
                    else self._click_found_image(hwnd, image_name, SUMMER_SCREEN_TIMEOUT, stop_event)
                )
                if click is None:
                    self._spam_back_until_gone(hwnd, stop_event)
                    return False
                if self._checkpoint(stop_event):
                    return False
                time.sleep(SETTLE_DELAY)
        return True

    def _reach_tournament_selected(self, hwnd, stop_event: threading.Event, tournament_type: str) -> bool:
        """Lobby -> Play -> Tournament -> type card, as one restartable unit --
        Tournament's equivalent of _reach_map_selected. Tournament is reached
        through Play just like Story/Raid (its nav_tournament button sits on the
        same gamemode menu, picked instead of Story), NOT via its own lobby
        entry the way Event is -- so this clicks Play first, then nav_tournament,
        then the chosen type's card (e.g. Solo Tournament). There's no map
        carousel or difficulty picker: picking the type IS the whole selection.
        On any failure it backs out to the lobby (_spam_back_until_gone) so the
        next attempt starts clean, same as the map/event paths do. The
        nav_entertournament confirm + solo Start tail is shared (see
        _enter_selected_stage)."""
        type_images = TOURNAMENT_TYPE_IMAGES.get(tournament_type)
        if type_images is None:
            self._log(f'[Macro] Unknown Tournament type "{tournament_type}" -- '
                       f'expected one of {TOURNAMENT_TYPE_ORDER}.')
            return False
        if isinstance(type_images, str):
            type_images = (type_images,)

        # Same "gamemode menu already open" shortcut as _reach_map_selected: if
        # nav_back is on screen the menu's already up, so checking for the lobby
        # and clicking Play would just burn LOBBY_CHECK_TIMEOUT waiting for a
        # Play button that doesn't exist there. Skip straight to nav_tournament.
        try:
            already_open = vision.find_image(hwnd, "nav_back") is not None
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        if already_open:
            self._log("[Macro] Already on the gamemode menu -- skipping the lobby and Play.")
        else:
            if not self._ensure_lobby(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False
            if not self._click_play(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False

        # nav_tournament: the Tournament button on the gamemode menu (picked in
        # place of Story), then its type card. Both clicks are wait-then-click
        # with a focus-safe verify via _click_found_image, and each screen
        # animates in, so a short settle follows before searching the next one.
        self._set_status(action="Clicking Tournament...")
        if self._click_found_image(hwnd, "nav_tournament", TOURNAMENT_SCREEN_TIMEOUT, stop_event) is None:
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        if self._checkpoint(stop_event):
            return False
        time.sleep(SETTLE_DELAY)

        self._set_status(action=f"Selecting {tournament_type}...")
        if self._click_found_image(hwnd, type_images[0], TOURNAMENT_SCREEN_TIMEOUT, stop_event) is None:
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        # Let the Enter Tournament screen finish animating in before the shared
        # tail searches for nav_entertournament (same reason the event/map
        # paths settle after their final click).
        time.sleep(SETTLE_DELAY)
        return not self._checkpoint(stop_event)

    def _reach_tower_selected(self, hwnd, stop_event: threading.Event, task: dict) -> bool:
        """Lobby -> Play -> Tower -> optional Traitless, as one restartable unit."""
        # Same "gamemode menu already open" shortcut as _reach_map_selected and
        # _reach_tournament_selected: if nav_back is on screen the menu's already
        # up, so checking for the lobby and clicking Play would just burn
        # LOBBY_CHECK_TIMEOUT waiting for a Play button that doesn't exist there.
        try:
            already_open = vision.find_image(hwnd, "nav_back") is not None
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] {exc}")
            return False
        if already_open:
            self._log("[Macro] Already on the gamemode menu -- skipping the lobby and Play.")
        else:
            if not self._ensure_lobby(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False
            if not self._click_play(hwnd, stop_event):
                return False
            if self._checkpoint(stop_event):
                return False

        self._set_status(action="Clicking Tower...")
        if self._click_found_image(hwnd, "nav_tower", TOWER_SCREEN_TIMEOUT, stop_event) is None:
            self._spam_back_until_gone(hwnd, stop_event)
            return False
        if self._checkpoint(stop_event):
            return False
        time.sleep(SETTLE_DELAY)

        if task.get("tower_mode") == "traitless":
            self._set_status(action="Selecting Traitless...")
            if self._click_found_image(hwnd, "Traitless_Tower", TOWER_SCREEN_TIMEOUT, stop_event) is None:
                self._spam_back_until_gone(hwnd, stop_event)
                return False
            self._log("[Macro] Traitless Tower selected.")
            if self._click_found_image(hwnd, "Floor", TOWER_SCREEN_TIMEOUT, stop_event, TOWER_CARD_REGION) is None:
                self._spam_back_until_gone(hwnd, stop_event)
                return False
            self._log("[Macro] Found Floor card -- waiting for select stage...")
        else:
            # Ensure that normal mode is actually selected so that we don't go off to some random place.
            self._set_status(action="Selecting Normal...")
            if self._click_found_image(hwnd, "normal_tower", TOWER_SCREEN_TIMEOUT, stop_event) is None:
                self._spam_back_until_gone(hwnd, stop_event)
                return False
            time.sleep(SETTLE_DELAY)
            self._log("[Macro] Normal Tower selected.")
            if self._click_found_image(hwnd, "Floor", TOWER_SCREEN_TIMEOUT, stop_event, region=TOWER_CARD_REGION) is None:
                self._spam_back_until_gone(hwnd, stop_event)
                return False
            self._log("[Macro] Found Floor card -- waiting for select stage...")
        time.sleep(SETTLE_DELAY)
        return not self._checkpoint(stop_event)

    def _scroll_find_and_click(self, hwnd, names, stop_event: threading.Event,
                                 scroll_power: int = None, scroll_nudges: int = None,
                                 label: str = "card") -> bool:
        """Wheel-scroll search for a card that may sit below the fold, modeled
        on (and reusing the tuning of) the Story map carousel search -- see
        stage_select.find_and_click_map. `names` is a tuple of candidate crops;
        any match wins, so a card with several visual states (e.g. Event Act 4,
        which has a peeking-in crop and a fully-scrolled-in one) is matched in
        whichever state it's currently showing.

        Checks what's on screen first (no scroll), so a card that's already
        visible is clicked without scrolling at all; otherwise nudges the list
        forward up to scroll_nudges times, then resets to the top and tries the
        whole pass again, up to MAX_PASSES times -- same recover-by-resetting
        approach the map carousel uses so a nudge that lands mid-animation
        doesn't strand the search. Returns whether it found and clicked one.
        """
        power = scroll_power if scroll_power is not None else stage_select.DEFAULT_SCROLL_POWER
        nudges = scroll_nudges if scroll_nudges is not None else stage_select.SCROLL_NUDGES_PER_PASS
        scroll_step = -120 * max(1, int(power))
        nudges = max(0, int(nudges))
        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        cx, cy = left + stage_select.SCROLL_CENTER[0], top + stage_select.SCROLL_CENTER[1]

        for pass_no in range(1, stage_select.MAX_PASSES + 1):
            if self._checkpoint(stop_event):
                return False
            self._log(f'[Macro] Looking for {label} (pass {pass_no}/{stage_select.MAX_PASSES}, '
                       f'up to {nudges} scrolls)...')
            for nudge in range(nudges + 1):
                if stop_event is not None and stop_event.is_set():
                    return False
                try:
                    match, found = vision.find_image_any(hwnd, tuple(names))
                except vision.TemplateNotFound as exc:
                    self._log(f"[Macro] {exc}")
                    return False
                if match is not None:
                    debug_path = self._debug_save(hwnd, found, match)
                    suffix = f" Debug: {debug_path}" if debug_path else ""
                    self._log(f'[Macro] Found "{found}" (score {match["score"]:.2f}) -- clicking it.{suffix}')
                    vision.click_match(self._mouse, hwnd, match)
                    return True

                if nudge < nudges:
                    self._mouse.move_to(cx, cy)
                    stage_select._hover_wiggle(self._mouse)
                    self._mouse.scroll(scroll_step)
                    time.sleep(stage_select.SETTLE_DELAY)
            # Not on this pass -- scroll all the way back to a known start
            # before trying again (see the map carousel's identical reset).
            self._log(f"[Macro] {label} not found in this pass -- scrolling back to the start.")
            self._mouse.move_to(cx, cy)
            stage_select._hover_wiggle(self._mouse)
            for _ in range(stage_select.SCROLL_RESET_NOTCHES):
                self._mouse.scroll(-scroll_step)
            time.sleep(stage_select.SETTLE_DELAY)

        self._log(f'[Macro] {label} never came into view after {stage_select.MAX_PASSES} scroll passes -- '
                   f'stopping. If its card was visibly scrolling by, its crop isn\'t matching your setup '
                   f'-- add your own via Settings > General > Image Manager.')
        return False

    def _spam_back_until_gone(self, hwnd, stop_event: threading.Event) -> None:
        # A failed map search can leave the run sitting on any of several
        # nested screens (mid-carousel-scroll, a map's detail panel, ...) --
        # repeatedly clicking Back until it's no longer found backs out
        # through however many of those there actually are, instead of
        # leaving the game wherever the failed search happened to stop for
        # the next run to trip over. Best-effort: not fatal either way, this
        # only ever runs after the run has already decided to give up.
        self._log("[Macro] Backing out after failed map search...")
        self._set_status(action="Backing out...")
        for attempt in range(1, BACK_SPAM_MAX_CLICKS + 1):
            if stop_event.is_set():
                return
            # wait_for_image, not a one-shot find_image: right after a
            # click, the next nested screen (if any) is still mid-
            # transition for a moment, and a single check timed unluckily
            # can miss a Back button that's genuinely there a beat later --
            # exactly what left this only clicking once when a second click
            # was still needed.
            try:
                match = vision.wait_for_image(
                    hwnd, "nav_back", timeout=BACK_SPAM_CHECK_TIMEOUT, stop_event=stop_event)
            except vision.TemplateNotFound as exc:
                self._log(f"[Macro] {exc}")
                return
            if match is None:
                self._log(f"[Macro] Back button gone after {attempt - 1} click(s) -- done.")
                return
            vision.click_match(self._mouse, hwnd, match)
            time.sleep(BACK_SPAM_DELAY)
        self._log(f"[Macro] Stopped backing out after {BACK_SPAM_MAX_CLICKS} clicks (Back button still found).")

    def _dismiss_party_overlay(self, hwnd, stop_event: threading.Event) -> bool:
        """Clear an optional party prompt before gamemode clicks.

        These prompts disappear quickly enough that a normal held click can
        release over the newly exposed card underneath. Use an immediate
        down/up and park away from the menu before checking that it cleared.
        """
        def _dismiss_target():
            match, name = vision.find_image_any(hwnd, PARTY_OVERLAY_IMAGE_NAMES)
            if match is None or name != "invite_players_open":
                return match, name

            # The outgoing player browser has a generic-looking Close
            # button. Only accept it after its unique title is visible so
            # another modal's Close button can never become this recovery's
            # click target.
            close = vision.find_image(hwnd, "invite_players_close")
            return (close, "invite_players_close") if close is not None else (None, name)

        try:
            match, name = _dismiss_target()
        except vision.TemplateNotFound:
            return True
        if name == "invite_players_open" and match is None:
            self._log('[Macro] "Invite Players" is open, but its Close button could not be found -- stopping.')
            return False
        if match is None:
            return True

        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        park_x, park_y = self._cxy("unit_info_reset")
        for attempt in range(1, 3):
            debug_path = self._debug_save(hwnd, name, match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f"[Macro] Found party overlay (score {match['score']:.2f}) -- "
                       f"dismissing it (attempt {attempt}/2).{suffix}")
            if not wm.activate_window(hwnd):
                self._log("[Macro] Couldn't confirm focus before dismissing the party overlay.")
            click_x, click_y = vision.ref_to_screen(hwnd, match["cx"], match["cy"])
            self._mouse.click(click_x, click_y, hold=0.0)
            self._mouse.move_to(left + park_x, top + park_y)
            if stop_event.is_set():
                return False
            time.sleep(0.3)
            try:
                remaining, name = _dismiss_target()
            except vision.TemplateNotFound:
                return True
            if name == "invite_players_open" and remaining is None:
                self._log('[Macro] "Invite Players" is still open, but its Close button could not be found -- stopping.')
                return False
            if remaining is None:
                return True
            match = remaining

        self._log("[Macro] Party overlay still showing after 2 dismissal attempts -- stopping.")
        return False

    def _click_gamemode_target(self, hwnd, stop_event: threading.Event, label: str, click_target) -> bool:
        """Click a gamemode card, recovering if the click opens an invite UI."""
        for attempt in range(1, GAMEMODE_OVERLAY_RETRY_ATTEMPTS + 1):
            click_target()
            if stop_event.is_set():
                return False
            time.sleep(GAMEMODE_OVERLAY_CHECK_DELAY)
            try:
                overlay, _ = vision.find_image_any(hwnd, PARTY_OVERLAY_IMAGE_NAMES)
            except vision.TemplateNotFound:
                return True
            if overlay is None:
                return True

            self._log(f"[Macro] A party overlay appeared after clicking {label} -- "
                       "closing it before retrying the intended card.")
            if not self._dismiss_party_overlay(hwnd, stop_event):
                return False
            if attempt < GAMEMODE_OVERLAY_RETRY_ATTEMPTS:
                self._log(f"[Macro] Retrying {label} "
                           f"(attempt {attempt + 1}/{GAMEMODE_OVERLAY_RETRY_ATTEMPTS}).")

        self._log(f"[Macro] Couldn't click {label} without reopening a party overlay -- stopping.")
        return False

    def _dismiss_afk_chamber(self, hwnd, last_clicked_at: float) -> float:
        """Click out of the AFK Chamber if the run has been parked in it.

        An Expedition encounter node can drop the client into the AFK Chamber.
        Nothing about that looks like a disconnect or a lobby, so the runner
        used to keep polling a screen where Victory/Defeat can never appear
        until MATCH_RESULT_TIMEOUT, once per node, for the rest of the run.

        Optional, like nav_disband/max_placement_reached: no afk_chamber.png
        means the check silently does nothing rather than failing a run.

        Returns the timestamp to carry into the next poll. Rate-limited,
        because the banner stays up for a moment after the exit is clicked and
        re-clicking every poll would fight the transition it just started.
        """
        if time.time() - last_clicked_at < AFK_CHAMBER_CLICK_COOLDOWN:
            return last_clicked_at
        try:
            match = vision.find_image(hwnd, "afk_chamber", region=AFK_CHAMBER_REGION)
        except vision.TemplateNotFound:
            return last_clicked_at
        if match is None:
            return last_clicked_at

        left, top, _, _ = wm.get_window_rect_screen(hwnd)
        self._log(f'[Macro] In the AFK Chamber (score {match["score"]:.2f}) -- clicking out of it.')
        self._set_status(action="Leaving the AFK Chamber...")
        self._mouse.click(left + AFK_CHAMBER_EXIT_CLICK[0], top + AFK_CHAMBER_EXIT_CLICK[1])
        return time.time()

    def _find_gamemode_card(self, hwnd, stop_event: threading.Event, names, label: str):
        """Locate a gamemode card, widening the search before giving up.

        The cards panel is normally boxed to GAMEMODE_CARD_REGION so the left
        3D viewport (player silhouettes, party [+] buttons) can't false-match.
        That box assumes a fixed layout, and the menu keeps growing -- Tower
        and Event cards landed in v0.19.0 -- so a card can now render outside
        it and the whole task fails with "never showed up".

        A boxed miss falls back to the full window instead. Template matching
        still only accepts a match by score (vision.DEFAULT_THRESHOLD), so the
        wider scan costs a few seconds, not accuracy -- the same reasoning
        that moved nav_play off its own fixed region.
        """
        try:
            match, name = vision.wait_for_image_any(
                hwnd, names, region=GAMEMODE_CARD_REGION,
                timeout=GAMEMODE_CLICK_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't find {label}: {exc}")
            return None, None
        if match is not None or stop_event.is_set():
            return match, name

        self._log(f'[Macro] {label} not in the cards panel within '
                   f'{GAMEMODE_CLICK_TIMEOUT:.0f}s -- widening to the whole window.')
        try:
            return vision.wait_for_image_any(
                hwnd, names, timeout=GAMEMODE_CARD_WIDE_TIMEOUT, stop_event=stop_event)
        except vision.TemplateNotFound as exc:
            self._log(f"[Macro] Can't find {label}: {exc}")
            return None, None

    def _click_gamemode(self, hwnd, stop_event: threading.Event, mode: str, wait_for_menu: bool = True) -> bool:
        # Story's card position doesn't move once the menu is open, so it's
        # just a fixed coordinate (see STORY_CLICK's comment). Raid's isn't
        # known as a fixed spot, so it's found by image search instead --
        # nav_back's appearance is only used to confirm the menu has
        # actually finished opening before either click fires, not to
        # locate Story itself. wait_for_menu=False skips that check
        # when the caller already confirmed nav_back is on screen (see the
        # "already open" shortcut in _run) -- no point polling for something
        # already known to be there.
        if wait_for_menu:
            match = None
            for attempt in range(1, PLAY_CLICK_RETRY_ATTEMPTS + 1):
                self._log(f'[Macro] Waiting for the gamemode menu to open (searching "nav_back", up to '
                           f'{STORY_SCREEN_TIMEOUT:.0f}s)...' if attempt == 1 else
                           f'[Macro] "nav_back" not found within {STORY_SCREEN_TIMEOUT:.0f}s -- still on '
                           f'the lobby, re-clicking Play (attempt {attempt}/{PLAY_CLICK_RETRY_ATTEMPTS})...')
                self._set_status(action='Waiting for gamemode menu ("nav_back")...')
                try:
                    match = vision.wait_for_image(
                        hwnd, "nav_back", timeout=STORY_SCREEN_TIMEOUT, stop_event=stop_event)
                except vision.TemplateNotFound as exc:
                    self._log(f"[Macro] Can't confirm the menu opened: {exc}")
                    return False
                if match is not None or stop_event.is_set():
                    break
                if attempt < PLAY_CLICK_RETRY_ATTEMPTS:
                    # Still on the lobby -- the earlier Play click plausibly
                    # never actually registered (see PLAY_CLICK_RETRY_ATTEMPTS'
                    # comment). Re-clicking is retriable in a way waiting
                    # even longer for a click that already failed isn't.
                    #
                    # This is also the first moment there is EVIDENCE that
                    # something is covering the lobby: Play matched, was
                    # clicked, and the menu still did not open. Clearing an
                    # overlay here rather than before every Play click means
                    # the normal path is untouched and only a demonstrably
                    # failed click pays for the extra search.
                    self._dismiss_lobby_overlay(hwnd)
                    if not self._click_play(hwnd, stop_event):
                        return False
            if match is None:
                if not stop_event.is_set():
                    self._log(f'[Macro] "nav_back" not found within {STORY_SCREEN_TIMEOUT:.0f}s x '
                               f'{PLAY_CLICK_RETRY_ATTEMPTS} attempt(s) -- the gamemode menu never opened '
                               f'(if it\'s visibly open, add your own crop of its Back button via '
                               f'Settings > General > Image Manager). Stopping.')
                    self._save_debug_screenshot_unconditional(hwnd, "gamemode_menu_timeout")
                return False

        if not self._dismiss_party_overlay(hwnd, stop_event):
            return False

        if mode == "boss_rush":
            self._log("[Macro] Menu open -- searching for Boss Rush...")
            self._set_status(action="Clicking Boss Rush...")
            match, name = self._find_gamemode_card(hwnd, stop_event, ("boss_rush",), "Boss Rush")
            if match is None:
                if not stop_event.is_set():
                    self._log(f'[Macro] "boss_rush" not found within {GAMEMODE_CLICK_TIMEOUT:.0f}s -- '
                              "the Boss Rush card never showed up, stopping.")
                return False
            debug_path = self._debug_save(hwnd, name, match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f"[Macro] Found Boss Rush (score {match['score']:.2f}) -- clicking it.{suffix}")
            clicked = self._click_gamemode_target(
                hwnd, stop_event, "Boss Rush", lambda: vision.click_match(self._mouse, hwnd, match))
            if not clicked or self._checkpoint(stop_event):
                return False
            self._log(f"[Boss Rush] Waiting {BOSS_RUSH_TRANSITION_DELAY:.0f}s for the map list to load.")
            self._interruptible_sleep(BOSS_RUSH_TRANSITION_DELAY, stop_event)
            return not self._checkpoint(stop_event)

        if mode == "expedition":
            self._log("[Macro] Menu open -- searching for Expedition...")
            self._set_status(action="Clicking Expedition...")
            match, name = self._find_gamemode_card(
                hwnd, stop_event, EXPEDITION_IMAGE_NAMES, "Expedition")
            if match is None:
                if not stop_event.is_set():
                    self._log(f'[Macro] "expedition" not found within {GAMEMODE_CLICK_TIMEOUT:.0f}s -- the '
                               f'Expedition card never showed up, stopping.')
                return False
            debug_path = self._debug_save(hwnd, name, match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f"[Macro] Found Expedition (score {match['score']:.2f}) -- clicking it.{suffix}")
            return self._click_gamemode_target(
                hwnd, stop_event, "Expedition", lambda: vision.click_match(self._mouse, hwnd, match))

        if mode == "challenge":
            self._log("[Macro] Menu open -- searching for Challenge...")
            self._set_status(action="Clicking Challenge...")
            match, name = self._find_gamemode_card(
                hwnd, stop_event, CHALLENGE_IMAGE_NAMES, "Challenge")
            if match is None:
                if not stop_event.is_set():
                    self._log(f'[Macro] "challenge" not found within {GAMEMODE_CLICK_TIMEOUT:.0f}s -- the '
                               f'Challenge card never showed up, stopping.')
                return False
            debug_path = self._debug_save(hwnd, name, match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f"[Macro] Found Challenge (score {match['score']:.2f}) -- clicking it.{suffix}")
            return self._click_gamemode_target(
                hwnd, stop_event, "Challenge", lambda: vision.click_match(self._mouse, hwnd, match))

        if mode == "raid":
            self._log("[Macro] Menu open -- searching for Raid...")
            self._set_status(action="Clicking Raid...")
            match, name = self._find_gamemode_card(
                hwnd, stop_event, RAID_IMAGE_NAMES, "Raid")
            if match is None:
                if not stop_event.is_set():
                    self._log(f'[Macro] "raid" not found within {GAMEMODE_CLICK_TIMEOUT:.0f}s -- the '
                               f'Raid card never showed up, stopping.')
                return False
            debug_path = self._debug_save(hwnd, name, match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f"[Macro] Found Raid (score {match['score']:.2f}) -- clicking it.{suffix}")
            return self._click_gamemode_target(
                hwnd, stop_event, "Raid", lambda: vision.click_match(self._mouse, hwnd, match))

        # story.png alone used to not be distinct enough to match reliably
        # (see STORY_CLICK's comment) -- Assets/ui/story/ holds a second
        # reference crop of the same card (every image in the folder is
        # tried per search, see vision.template_variant_paths), so image
        # search is worth trying again here before falling back to the
        # fixed coordinate.
        self._log("[Macro] Menu open -- searching for Story...")
        self._set_status(action="Clicking Story...")
        match, name = self._find_gamemode_card(hwnd, stop_event, STORY_IMAGE_NAMES, "Story")
        if match is not None:
            debug_path = self._debug_save(hwnd, name, match)
            suffix = f" Debug: {debug_path}" if debug_path else ""
            self._log(f"[Macro] Found Story (score {match['score']:.2f}) -- clicking it.{suffix}")
            clicked = self._click_gamemode_target(
                hwnd, stop_event, "Story", lambda: vision.click_match(self._mouse, hwnd, match))
        else:
            if stop_event.is_set():
                return False
            story_x, story_y = self._cxy("story_click")
            self._log(f"[Macro] Story card not found by image search -- falling back to fixed coordinate ({story_x}, {story_y}).")
            left, top, _, _ = wm.get_window_rect_screen(hwnd)
            clicked = self._click_gamemode_target(
                hwnd, stop_event, "Story", lambda: self._mouse.click(left + story_x, top + story_y))
        if not clicked:
            return False
        # Unlike Raid/Expedition/Challenge (which wait_for_image their own
        # gamemode card before clicking, naturally giving the screen a
        # moment), the fixed-coordinate fallback above is blind -- nav_back
        # confirms the MENU is open, not that the map carousel it leads to
        # has finished rendering. The very first carousel scan used to fire
        # immediately after this click, which could catch it mid-transition
        # and misread/misclick the wrong card (reported: landing on the
        # wrong map, e.g. a different Story map instead of the one asked
        # for).
        time.sleep(SETTLE_DELAY)
        return True
