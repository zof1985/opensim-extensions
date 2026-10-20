import numpy as np
import pytest

opensim = pytest.importorskip("opensim")

from opensim_extensions import save_animation
from opensim_extensions.models.user.user import DEFAULT_MESHES_DIR, DEFAULT_MODEL_PATH
from opensim_extensions.opensim import register_geometry_search_path


def make_model():
    register_geometry_search_path(opensim, DEFAULT_MESHES_DIR)
    model = opensim.Model(str(DEFAULT_MODEL_PATH))
    model.initSystem()
    return model


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


def make_motion():
    table = opensim.TimeSeriesTable()
    table.setColumnLabels(["/jointset/hip_r/hip_flexion_r/value"])
    table.addTableMetaDataString("inDegrees", "yes")
    table.appendRow(1.0, opensim.RowVector([0.0]))
    table.appendRow(1.5, opensim.RowVector([90.0]))
    return table


def test_save_animation_renders_sto_offscreen_and_restores_pose(
    monkeypatch, tmp_path
):
    import imageio
    import opensim_extensions.viewer.visualizer as visualizer_module

    model = make_model()
    raw_model = model
    state = model.getWorkingState()
    coordinate = raw_model.getCoordinateSet().get("hip_flexion_r")
    original_value = coordinate.getValue(state)
    writer = FakeWriter()
    writer_options = {}
    rendered_values = []
    visualizer_options = {}
    camera_options = {}
    visibility_options = {}

    class FakeVisualizer:
        def __init__(self, visualizer_model, **kwargs):
            visualizer_options.update(kwargs)
            self.model = visualizer_model

        def show(self, render_state):
            rendered_values.append(
                self.model.model.getCoordinateSet()
                .get("hip_flexion_r")
                .getValue(render_state)
            )

        def capture_frame(self):
            return np.zeros((4, 6, 3), dtype=np.uint8)

        def set_camera(self, **kwargs):
            camera_options.update(kwargs)

        def set_ground_visible(self, visible):
            visibility_options["ground"] = visible

        def set_muscles_visible(self, visible):
            visibility_options["muscles"] = visible

        def set_markers_visible(self, visible):
            visibility_options["markers"] = visible

        def set_axes_visible(self, visible):
            visibility_options["axes"] = visible

        def close(self):
            pass

    def fake_get_writer(*args, **kwargs):
        writer_options.update(kwargs)
        return writer

    monkeypatch.setattr(visualizer_module, "VTKVisualizer", FakeVisualizer)
    monkeypatch.setattr(imageio, "get_writer", fake_get_writer)

    destination = tmp_path / "nested" / "motion.mp4"
    result = save_animation(
        model,
        make_motion(),
        destination,
        size=(640, 360),
        camera_position=(2.0, 1.0, 0.0),
        camera_focal_point=(0.0, 1.0, 0.0),
        camera_view_up=(0.0, 1.0, 0.0),
        show_ground=False,
        show_muscles=True,
        show_markers=False,
        show_axes=True,
    )

    assert result == destination
    assert destination.parent.is_dir()
    assert writer_options["fps"] == pytest.approx(2.0)
    assert len(writer.frames) == 2
    assert rendered_values == pytest.approx([0.0, np.pi / 2])
    assert visualizer_options == {"offscreen": True, "size": (640, 360)}
    assert camera_options == {
        "position": (2.0, 1.0, 0.0),
        "focal_point": (0.0, 1.0, 0.0),
        "view_up": (0.0, 1.0, 0.0),
    }
    assert visibility_options == {
        "ground": False,
        "muscles": True,
        "markers": False,
        "axes": True,
    }
    assert writer.closed
    assert coordinate.getValue(state) == pytest.approx(original_value)


def test_save_animation_accepts_sto_path(monkeypatch, tmp_path):
    import imageio
    import opensim_extensions.viewer.visualizer as visualizer_module

    model = make_model()
    table = make_motion()
    sto_path = tmp_path / "motion.sto"
    opensim.STOFileAdapter.write(table, str(sto_path))

    monkeypatch.setattr(
        visualizer_module,
        "VTKVisualizer",
        lambda *args, **kwargs: type(
            "FakeVisualizer",
            (),
            {
                "show": lambda self, state: None,
                "capture_frame": lambda self: np.zeros((2, 2, 3), dtype=np.uint8),
                "set_ground_visible": lambda self, visible: None,
                "set_muscles_visible": lambda self, visible: None,
                "set_markers_visible": lambda self, visible: None,
                "set_axes_visible": lambda self, visible: None,
                "close": lambda self: None,
            },
        )(),
    )
    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: FakeWriter())

    assert save_animation(model, sto_path, tmp_path / "motion.mp4").name == "motion.mp4"


def test_save_animation_rejects_table_without_matching_coordinates(tmp_path):
    model = make_model()
    table = opensim.TimeSeriesTable()
    table.setColumnLabels(["not_a_coordinate"])
    table.appendRow(0.0, opensim.RowVector([0.0]))
    table.appendRow(1.0, opensim.RowVector([1.0]))

    with pytest.raises(ValueError, match="no columns matching"):
        save_animation(model, table, tmp_path / "motion.mp4")


def test_save_animation_rejects_invalid_size(tmp_path):
    model = make_model()

    with pytest.raises(ValueError, match="positive integers"):
        save_animation(model, make_motion(), tmp_path / "motion.mp4", size=(640, 0))

    with pytest.raises(ValueError, match="must both be even"):
        save_animation(model, make_motion(), tmp_path / "motion.mp4", size=(641, 360))


def test_save_animation_requires_mp4_path(tmp_path):
    model = make_model()

    with pytest.raises(ValueError, match=".mp4 extension"):
        save_animation(model, make_motion(), tmp_path / "motion.avi")


def test_save_animation_rejects_invalid_camera_vector(tmp_path):
    model = make_model()

    with pytest.raises(ValueError, match="camera_position"):
        save_animation(
            model,
            make_motion(),
            tmp_path / "motion.mp4",
            camera_position=(1.0, 2.0),
        )


def test_save_animation_rejects_non_boolean_visibility_option(tmp_path):
    model = make_model()

    with pytest.raises(TypeError, match="show_ground must be a boolean"):
        save_animation(model, make_motion(), tmp_path / "motion.mp4", show_ground="yes")
