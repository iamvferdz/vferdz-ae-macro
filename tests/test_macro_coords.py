import main


def test_team_button_coordinate_override_can_be_saved_and_cleared(monkeypatch):
    state = {}
    monkeypatch.setattr(main.cfg, "load", lambda: dict(state))

    def update(patch):
        state.update(patch)
        return dict(state)

    monkeypatch.setattr(main.cfg, "update", update)
    api = object.__new__(main.Api)

    assert api.set_macro_coords({"team_button_x": 438, "team_button_y": 570})["saved"] == [
        "team_button_x", "team_button_y"
    ]
    coords = api.get_macro_coords()
    assert coords["team_button_x"] == 438
    assert coords["team_button_y"] == 570

    assert api.clear_macro_coord("team_button")["ok"] is True
    coords = api.get_macro_coords()
    assert coords["team_button_x"] is None
    assert coords["team_button_y"] is None
    assert api.clear_macro_coord("screen_middle")["ok"] is False


def test_boss_rush_card_coordinates_can_be_saved_and_read_back(monkeypatch):
    state = {}
    monkeypatch.setattr(main.cfg, "load", lambda: dict(state))

    def update(patch):
        state.update(patch)
        return dict(state)

    monkeypatch.setattr(main.cfg, "update", update)
    api = object.__new__(main.Api)

    changes = {
        "boss_rush_card_left_x": 220,
        "boss_rush_card_left_y": 340,
        "boss_rush_card_middle_x": 570,
        "boss_rush_card_middle_y": 345,
        "boss_rush_card_right_x": 910,
        "boss_rush_card_right_y": 350,
    }
    assert api.set_macro_coords(changes)["saved"] == list(changes)
    assert api.set_macro_coord("boss_rush_card_right_x", 920)["ok"] is True
    changes["boss_rush_card_right_x"] = 920
    assert {key: api.get_macro_coords()[key] for key in changes} == changes
    assert all(key in main.MACRO_COORD_DEFAULTS for key in changes)


def test_old_default_team_loadout_coordinates_migrate_without_overwriting_custom_points(monkeypatch):
    state = {
        "team_loadout_x": 800,
        "team_loadout_y": 324,
        "team_loadout_row_height": 126,
    }
    monkeypatch.setattr(main.cfg, "load", lambda: dict(state))
    api = object.__new__(main.Api)

    coords = api.get_macro_coords()
    assert (coords["team_loadout_x"], coords["team_loadout_y"],
            coords["team_loadout_row_height"]) == (270, 241, 38)

    state["team_loadout_x"] = 805
    coords = api.get_macro_coords()
    assert (coords["team_loadout_x"], coords["team_loadout_y"],
            coords["team_loadout_row_height"]) == (805, 241, 38)
