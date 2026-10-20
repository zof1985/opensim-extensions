from __future__ import annotations

import pytest

opensim = pytest.importorskip("opensim")

from opensim_extensions.viewer import VTKViewer, viewer as viewer_module
from opensim_extensions.framework import Framework


def make_model() -> opensim.Model:
    model = opensim.Model()
    body = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    model.addBody(body)
    model.addJoint(
        opensim.WeldJoint(
            "root",
            model.getGround(),
            opensim.Vec3(0),
            opensim.Vec3(0),
            body,
            opensim.Vec3(0),
            opensim.Vec3(0),
        )
    )
    model.initSystem()
    return model


def test_viewer_accepts_native_models_without_opening_a_window(monkeypatch):
    model = make_model()
    captured = {}
    expected = object()

    def fake_start_player(adapter, motion, *, loop, fps):
        captured.update(adapter=adapter, motion=motion, loop=loop, fps=fps)
        return expected

    monkeypatch.setattr(viewer_module, "start_player", fake_start_player)

    result = VTKViewer().show(model, loop=True, fps=24.0)

    assert result is expected
    assert captured["adapter"].model is model
    assert captured["adapter"].state.getTime() == pytest.approx(
        model.getWorkingState().getTime()
    )
    assert captured["loop"] is True
    assert captured["fps"] == 24.0


def test_viewer_accepts_framework_and_displays_merged_snapshot(monkeypatch):
    model = make_model()
    framework = Framework({"member": model})
    captured = {}

    def fake_start_player(adapter, motion, *, loop, fps):
        captured["model"] = adapter.model
        return "viewer"

    monkeypatch.setattr(viewer_module, "start_player", fake_start_player)

    assert VTKViewer().show(framework) == "viewer"
    assert isinstance(captured["model"], opensim.Model)
    assert captured["model"] is not model
    assert captured["model"].getBodySet().getSize() == 1
    assert model.getBodySet().getSize() == 1


def test_viewer_rejects_values_outside_the_supported_input_types():
    with pytest.raises(TypeError, match="opensim.Model or Framework"):
        VTKViewer().show(object())


def make_pin_model() -> opensim.Model:
    model = opensim.Model()
    body = opensim.Body("body", 1.0, opensim.Vec3(0), opensim.Inertia(1))
    body.attachGeometry(opensim.Brick(opensim.Vec3(0.1, 0.2, 0.3)))
    model.addBody(body)
    model.addJoint(
        opensim.PinJoint(
            "joint",
            model.getGround(),
            opensim.Vec3(0),
            opensim.Vec3(0),
            body,
            opensim.Vec3(0),
            opensim.Vec3(0),
        )
    )
    model.initSystem()
    return model


def make_motion() -> opensim.TimeSeriesTable:
    table = opensim.TimeSeriesTable()
    table.setColumnLabels(["joint_coord_0"])
    table.appendRow(0.0, opensim.RowVector([0.0]))
    table.appendRow(1.0, opensim.RowVector([1.0]))
    return table


def test_save_snapshot_writes_vector_svg_for_model_and_framework(tmp_path):
    model = make_pin_model()
    viewer = VTKViewer()

    result = viewer.save_snapshot(model, path=tmp_path / "model.svg")
    viewer.save_snapshot(Framework({"member": model}), path=tmp_path / "fw.svg")

    assert result == tmp_path / "model.svg"
    for name in ("model.svg", "fw.svg"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert text.startswith("<?xml") and "<polygon" in text and "<image" not in text


def test_save_snapshot_rejects_non_svg_and_frame_without_motion(tmp_path):
    model = make_pin_model()
    viewer = VTKViewer()
    with pytest.raises(ValueError, match=".svg"):
        viewer.save_snapshot(model, path=tmp_path / "pose.png")
    with pytest.raises(ValueError, match="require a motion"):
        viewer.save_snapshot(model, frame=3, path=tmp_path / "pose.svg")
    with pytest.raises(TypeError):
        viewer.save_snapshot(object(), path=tmp_path / "pose.svg")


def test_save_snapshot_applies_motion_frame_or_time_without_mutating_model(tmp_path):
    model = make_pin_model()
    coordinate = model.getCoordinateSet().get(0)
    before = coordinate.getValue(model.getWorkingState())
    viewer = VTKViewer()
    motion = make_motion()

    viewer.save_snapshot(model, frame=0, path=tmp_path / "first.svg", motion=motion)
    viewer.save_snapshot(model, frame=1, path=tmp_path / "last.svg", motion=motion)
    viewer.save_snapshot(model, time_s=1.0, path=tmp_path / "time.svg", motion=motion)

    first = (tmp_path / "first.svg").read_text(encoding="utf-8")
    last = (tmp_path / "last.svg").read_text(encoding="utf-8")
    assert first != last
    assert last == (tmp_path / "time.svg").read_text(encoding="utf-8")
    assert coordinate.getValue(model.getWorkingState()) == before
    with pytest.raises(ValueError, match="out of range"):
        viewer.save_snapshot(model, frame=5, path=tmp_path / "x.svg", motion=motion)


def test_save_animation_delegates_to_the_renderer(monkeypatch, tmp_path):
    model = make_pin_model()
    captured = {}

    def fake_render(native, table, path, **options):
        captured.update(model=native, table=table, path=path, options=options)
        return path

    monkeypatch.setattr(viewer_module, "_render_animation", fake_render)
    motion = make_motion()

    result = VTKViewer().save_animation(
        model, tmp_path / "m.mp4", motion=motion, size=(640, 360)
    )

    assert result == tmp_path / "m.mp4"
    assert captured["model"] is model and captured["table"] is motion
    assert captured["options"] == {"size": (640, 360)}
