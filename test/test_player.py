import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from opensim_models import OpenSimModel

# DEFAULT_MODEL_PATH is User's internal default, reused here as a real .osim
# fixture with both a rotational (hip_flexion_r) and a translational
# (pelvis_tx) coordinate, to test MotionData's unit handling.
from opensim_models.models.user.user import DEFAULT_MODEL_PATH
from opensim_models._gui.player import MotionData, MotionPlayer

opensim = pytest.importorskip("opensim")


def make_model():
    return OpenSimModel(model_path=DEFAULT_MODEL_PATH)


def make_table(column_names, times, rows, *, in_degrees=True):
    table = opensim.TimeSeriesTable()
    table.setColumnLabels(column_names)
    if in_degrees:
        table.addTableMetaDataString("inDegrees", "yes")
    for time, row in zip(times, rows):
        table.appendRow(time, opensim.RowVector(row))
    return table


# ---------------------------------------------------------------------------
# MotionData
# ---------------------------------------------------------------------------


def test_motion_data_converts_rotational_columns_from_degrees():
    model = make_model()
    table = make_table(["hip_flexion_r"], [0.0, 1.0], [[0.0], [90.0]])

    data = MotionData(model, table)

    assert data.values_at(1.0)["hip_flexion_r"] == pytest.approx(np.radians(90.0))


def test_motion_data_leaves_translational_columns_in_metres():
    model = make_model()
    table = make_table(["pelvis_tx"], [0.0, 1.0], [[0.0], [90.0]])

    data = MotionData(model, table)

    # even though inDegrees=yes, a translational coordinate is never
    # converted: 90 here means 90 metres, not 90 degrees
    assert data.values_at(1.0)["pelvis_tx"] == pytest.approx(90.0)


def test_motion_data_does_not_convert_when_table_is_not_in_degrees():
    model = make_model()
    table = make_table(["hip_flexion_r"], [0.0, 1.0], [[0.0], [1.5]], in_degrees=False)

    data = MotionData(model, table)

    assert data.values_at(1.0)["hip_flexion_r"] == pytest.approx(1.5)


def test_motion_data_keeps_a_column_that_is_not_a_model_coordinate():
    model = make_model()
    table = make_table(["not_a_coordinate"], [0.0, 1.0], [[1.0], [2.0]])

    data = MotionData(model, table)

    assert data.values_at(0.5)["not_a_coordinate"] == pytest.approx(1.5)


def test_motion_data_interpolates_linearly_between_samples():
    model = make_model()
    table = make_table(["hip_flexion_r"], [0.0, 2.0], [[0.0], [20.0]])

    data = MotionData(model, table)

    assert data.values_at(1.0)["hip_flexion_r"] == pytest.approx(np.radians(10.0))


def test_motion_data_clamps_queries_outside_its_range():
    model = make_model()
    table = make_table(["hip_flexion_r"], [1.0, 2.0], [[10.0], [20.0]])

    data = MotionData(model, table)

    assert data.values_at(-5.0)["hip_flexion_r"] == pytest.approx(np.radians(10.0))
    assert data.values_at(50.0)["hip_flexion_r"] == pytest.approx(np.radians(20.0))


def test_motion_data_exposes_start_and_end_time():
    model = make_model()
    table = make_table(["hip_flexion_r"], [0.25, 3.5], [[0.0], [1.0]])

    data = MotionData(model, table)

    assert data.start_time == pytest.approx(0.25)
    assert data.end_time == pytest.approx(3.5)


def test_motion_data_rejects_fewer_than_two_rows():
    model = make_model()
    table = make_table(["hip_flexion_r"], [0.0], [[0.0]])

    with pytest.raises(ValueError, match="at least 2 rows"):
        MotionData(model, table)


# ---------------------------------------------------------------------------
# MotionPlayer
# ---------------------------------------------------------------------------


def test_motion_player_rejects_a_non_positive_time_span():
    with pytest.raises(ValueError, match="end_time must be greater than start_time"):
        MotionPlayer(1.0, 1.0)


def test_motion_player_starts_paused_at_the_beginning():
    player = MotionPlayer(0.0, 2.0)

    assert player.playing is False
    assert player.time == pytest.approx(0.0)
    assert player.speed == pytest.approx(1.0)
    assert player.fraction == pytest.approx(0.0)


def test_motion_player_play_pause_toggles():
    player = MotionPlayer(0.0, 2.0)

    player.play_pause()
    assert player.playing is True

    player.play_pause()
    assert player.playing is False


def test_motion_player_advance_is_a_noop_while_paused():
    player = MotionPlayer(0.0, 2.0)

    player.advance(1.0)

    assert player.time == pytest.approx(0.0)


def test_motion_player_advance_moves_time_by_dt_times_speed():
    player = MotionPlayer(0.0, 2.0)
    player.play_pause()

    player.advance(0.5)

    assert player.time == pytest.approx(0.5)
    assert player.fraction == pytest.approx(0.25)


def test_motion_player_advance_clamps_and_pauses_at_the_end_without_loop():
    player = MotionPlayer(0.0, 2.0, loop=False)
    player.play_pause()

    player.advance(5.0)

    assert player.time == pytest.approx(2.0)
    assert player.playing is False


def test_motion_player_advance_wraps_around_at_the_end_with_loop():
    player = MotionPlayer(0.0, 2.0, loop=True)
    player.play_pause()

    player.advance(2.5)

    assert player.time == pytest.approx(0.5)
    assert player.playing is True


def test_motion_player_advance_wraps_around_at_the_start_in_reverse_with_loop():
    player = MotionPlayer(0.0, 2.0, loop=True)
    player.seek_fraction(0.25)
    player.play_pause()
    player.fast_backward()  # -1x

    player.advance(1.0)

    assert player.time == pytest.approx(1.5)
    assert player.playing is True


def test_motion_player_stop_resets_time_speed_and_pauses():
    player = MotionPlayer(0.0, 2.0)
    player.play_pause()
    player.fast_forward()
    player.advance(1.0)

    player.stop()

    assert player.time == pytest.approx(0.0)
    assert player.playing is False
    assert player.speed == pytest.approx(1.0)


def test_motion_player_toggle_loop():
    player = MotionPlayer(0.0, 2.0, loop=False)

    player.toggle_loop()
    assert player.loop is True

    player.toggle_loop()
    assert player.loop is False


def test_motion_player_fast_forward_steps_speed_up_and_starts_playing():
    player = MotionPlayer(0.0, 2.0)

    player.fast_forward()
    assert player.speed == pytest.approx(2.0)
    assert player.playing is True

    player.fast_forward()
    assert player.speed == pytest.approx(4.0)


def test_motion_player_fast_forward_is_capped():
    player = MotionPlayer(0.0, 2.0)

    for _ in range(10):
        player.fast_forward()

    assert player.speed == pytest.approx(8.0)


def test_motion_player_fast_backward_steps_speed_down_and_starts_playing():
    player = MotionPlayer(0.0, 2.0)

    player.fast_backward()
    assert player.speed == pytest.approx(-1.0)
    assert player.playing is True

    player.fast_backward()
    assert player.speed == pytest.approx(-2.0)


def test_motion_player_fast_backward_is_capped():
    player = MotionPlayer(0.0, 2.0)

    for _ in range(10):
        player.fast_backward()

    assert player.speed == pytest.approx(-8.0)


def test_motion_player_fast_forward_after_fast_backward_steps_back_toward_forward():
    player = MotionPlayer(0.0, 2.0)
    player.fast_backward()
    player.fast_backward()  # -2x

    player.fast_forward()

    assert player.speed == pytest.approx(-1.0)


def test_motion_player_seek_fraction_sets_time_and_clamps():
    player = MotionPlayer(0.0, 2.0)

    player.seek_fraction(0.5)
    assert player.time == pytest.approx(1.0)

    player.seek_fraction(-1.0)
    assert player.time == pytest.approx(0.0)

    player.seek_fraction(2.0)
    assert player.time == pytest.approx(2.0)
