"""Vector SVG export of a VTK scene.

VTK's GL2PS exporter rasterises the polygonal props drawn by the OpenGL2
backend, so the scene is projected here with the active camera instead:
every triangle becomes an SVG polygon (painter's algorithm, flat shading)
and every polyline an SVG polyline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["write_svg"]

_LIGHT_DIRECTION = np.array([0.3, 0.4, 1.0]) / np.linalg.norm([0.3, 0.4, 1.0])


def _hex(color: np.ndarray) -> str:
    red, green, blue = (np.clip(color, 0.0, 1.0) * 255).round().astype(int)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _actor_geometry(vtk: Any, actor: Any) -> tuple[Any, Any] | None:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    mapper = actor.GetMapper()
    if mapper is None or not actor.GetVisibility():
        return None
    mapper.Update()
    data = mapper.GetInput()
    if data is None or data.GetNumberOfPoints() == 0:
        return None
    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(data)
    triangulator.PassLinesOn()
    triangulator.PassVertsOff()
    triangulator.Update()
    poly = triangulator.GetOutput()

    matrix = actor.GetMatrix()
    world = np.array([[matrix.GetElement(r, c) for c in range(4)] for r in range(4)])
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    homogeneous = np.hstack([points, np.ones((len(points), 1))]) @ world.T
    return poly, homogeneous


def write_svg(renderer: Any, window_size: tuple[int, int], path: str | Path) -> Path:
    """Write ``renderer``'s visible polygonal props as a vector SVG.

    Parameters
    ----------
    renderer : vtkRenderer
        Renderer whose actors and active camera define the picture.
    window_size : tuple[int, int]
        Picture size in pixels as ``(width, height)``.
    path : str or pathlib.Path
        Destination file.

    Returns
    -------
    pathlib.Path
        The destination path.
    """
    import vtk
    from vtkmodules.util.numpy_support import vtk_to_numpy

    width, height = window_size
    camera = renderer.GetActiveCamera()
    projection = camera.GetCompositeProjectionTransformMatrix(width / height, -1, 1)
    view = np.array([[projection.GetElement(r, c) for c in range(4)] for r in range(4)])
    view_matrix = camera.GetModelViewTransformMatrix()
    model_view = np.array([[view_matrix.GetElement(r, c) for c in range(4)] for r in range(4)])

    background = renderer.GetBackground()
    # (depth, svg element) pairs; larger depth is further from the camera.
    elements: list[tuple[float, str]] = []

    actors = renderer.GetActors()
    actors.InitTraversal()
    for _ in range(actors.GetNumberOfItems()):
        actor = actors.GetNextActor()
        geometry = _actor_geometry(vtk, actor)
        if geometry is None:
            continue
        poly, world_points = geometry
        clip = world_points @ view.T
        w = clip[:, 3:4]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        ndc = clip[:, :3] / w
        screen = np.column_stack(
            [(ndc[:, 0] + 1.0) / 2.0 * width, (1.0 - ndc[:, 1]) / 2.0 * height]
        )
        visible = (clip[:, 3] > 0) & (np.abs(ndc[:, 2]) <= 1.0)
        property_ = actor.GetProperty()
        base = np.array(property_.GetColor())
        opacity = property_.GetOpacity()
        opacity_attr = f' opacity="{opacity:.3f}"' if opacity < 1.0 else ""

        connectivity = vtk_to_numpy(poly.GetPolys().GetConnectivityArray())
        triangles = connectivity.reshape(-1, 3) if connectivity.size else np.empty((0, 3), int)
        if len(triangles):
            eye = (world_points @ model_view.T)[:, :3]
            tri_eye = eye[triangles]
            normals = np.cross(tri_eye[:, 1] - tri_eye[:, 0], tri_eye[:, 2] - tri_eye[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            lengths[lengths == 0] = 1.0
            shade = 0.35 + 0.65 * np.abs((normals / lengths[:, None]) @ _LIGHT_DIRECTION)
            keep = visible[triangles].all(axis=1)
            for index in np.flatnonzero(keep):
                polygon = screen[triangles[index]]
                text = " ".join(f"{x:.2f},{y:.2f}" for x, y in polygon)
                color = _hex(base * shade[index])
                elements.append(
                    (
                        float(ndc[triangles[index], 2].mean()),
                        f'<polygon points="{text}" fill="{color}" stroke="{color}" '
                        f'stroke-width="0.3"{opacity_attr}/>',
                    )
                )

        lines = poly.GetLines()
        if lines.GetNumberOfCells():
            offsets = vtk_to_numpy(lines.GetOffsetsArray())
            line_points = vtk_to_numpy(lines.GetConnectivityArray())
            stroke = _hex(base)
            line_width = max(float(property_.GetLineWidth()), 0.5)
            for start, stop in zip(offsets[:-1], offsets[1:]):
                ids = line_points[start:stop]
                if not visible[ids].all():
                    continue
                text = " ".join(f"{x:.2f},{y:.2f}" for x, y in screen[ids])
                elements.append(
                    (
                        float(ndc[ids, 2].mean()),
                        f'<polyline points="{text}" fill="none" stroke="{stroke}" '
                        f'stroke-width="{line_width:.2f}"{opacity_attr}/>',
                    )
                )

    elements.sort(key=lambda item: -item[0])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(element for _, element in elements)
    destination.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<rect width="100%" height="100%" fill="{_hex(np.array(background))}"/>\n'
        f"{body}\n</svg>\n",
        encoding="utf-8",
    )
    return destination
