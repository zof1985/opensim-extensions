from __future__ import annotations

__all__ = ["DEFAULT_DATASET", "AnthropometricReference", "load_ansur", "resolve_reference"]

from dataclasses import dataclass
from pathlib import Path
import csv
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

DEFAULT_DATASET = Path(__file__).resolve().parent / "assets" / "ansur_ref.csv"


@dataclass(frozen=True)
class AnthropometricReference:
    """Anthropometric values resolved at one ANSUR percentile.

    Parameters
    ----------
    gender : str
        Normalized sex code, either ``"M"`` or ``"F"``.
    percentile : float
        Percentile used to calculate all numeric measurements.
    values : dict[str, float]
        Measurement names mapped to values in their documented source units.
    """

    gender: str
    percentile: float
    values: dict[str, float]

    @property
    def height_m(self) -> float:
        """Return stature in metres.

        Returns
        -------
        float
            ANSUR stature in metres.
        """
        return self.values["stature_m"]

    @property
    def height_cm(self) -> float:
        """Return stature in centimetres.

        Returns
        -------
        float
            ANSUR stature in centimetres.
        """
        return self.height_m * 100.0


def _read_raw_csv(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))

    if not rows:
        raise ValueError(f"ANSUR dataset is empty: {path}")

    header = [name.strip() for name in rows[0] if name.strip()]
    if not header or header[0] != "Branch":
        raise ValueError("Unexpected ANSUR header: expected Branch as first column")

    expected = len(header) + 1
    data_rows = []
    for line_number, row in enumerate(rows[1:], start=2):
        if not row or not any(value.strip() for value in row):
            continue
        if len(row) != expected:
            raise ValueError(
                f"Invalid ANSUR row {line_number}: expected {expected} fields, got {len(row)}"
            )
        data_rows.append(row)

    return pd.DataFrame(data_rows, columns=["subject_id", *header])


def load_ansur(path: str | Path = DEFAULT_DATASET) -> pd.DataFrame:
    """Load ANSUR II while repairing its implicit leading subject_id column.

    Linear measurements are kept in their source units (mostly millimetres).
    Only the percentile reference exposes stature in metres, matching OpenSim.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to the ANSUR CSV file.

    Returns
    -------
    pandas.DataFrame
        Normalized ANSUR records with an explicit ``subject_id`` column.

    Raises
    ------
    ValueError
        If the file is empty, malformed, or contains unsupported values.
    """
    frame = _read_raw_csv(Path(path))
    frame["Gender"] = (
        frame["Gender"]
        .str.strip()
        .str.upper()
        .map({"MALE": "M", "FEMALE": "F", "M": "M", "F": "F"})
    )
    if frame["Gender"].isna().any():
        raise ValueError("ANSUR contains an unsupported gender value")

    numeric_columns = frame.columns.difference(
        ["Branch", "Component", "Gender", "BMI_class", "Height_class"]
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["stature_m"].isna().any():
        raise ValueError("ANSUR contains invalid stature_m values")
    return frame


def _pchip_with_linear_tails(x: np.ndarray, y: np.ndarray):
    """Build a PCHIP interpolator that extrapolates linearly, not cubically.

    Extending the fitted cubic polynomial beyond the data (SciPy's
    ``extrapolate=True``) can swing arbitrarily far from the trend for
    queries well outside the sampled range. Continuing instead along the
    tangent at each boundary keeps extrapolated values monotonic with the
    local trend.
    """
    interpolator = PchipInterpolator(x, y, extrapolate=False)
    derivative = interpolator.derivative()
    x_min, x_max = float(x[0]), float(x[-1])
    y_min, y_max = float(y[0]), float(y[-1])
    slope_min, slope_max = float(derivative(x_min)), float(derivative(x_max))

    def evaluate(query: float) -> float:
        if query < x_min:
            return y_min + slope_min * (query - x_min)
        if query > x_max:
            return y_max + slope_max * (query - x_max)
        return float(interpolator(query))

    return evaluate


def _height_percentile(statures_m: np.ndarray, height_cm: float) -> float:
    height_m = height_cm / 100.0
    sorted_statures = np.sort(statures_m)
    minimum, maximum = float(sorted_statures[0]), float(sorted_statures[-1])

    if minimum <= height_m <= maximum:
        # Empirical CDF: no subject selection, no extrapolation.
        return float(
            np.searchsorted(sorted_statures, height_m, side="right")
            * 100
            / len(sorted_statures)
        )

    warnings.warn(
        f"height={height_cm:g} cm is outside ANSUR range "
        f"[{minimum * 100:.1f}, {maximum * 100:.1f}] cm; "
        "extrapolating the stature percentile with a PCHIP interpolator.",
        category=UserWarning,
        stacklevel=2,
    )
    unique_statures = np.unique(sorted_statures)
    counts = np.searchsorted(sorted_statures, unique_statures, side="right")
    empirical_percentiles = counts * 100.0 / len(sorted_statures)
    return _pchip_with_linear_tails(unique_statures, empirical_percentiles)(height_m)


_EXCLUDED_COLUMNS = {
    "subject_id",
    "Branch",
    "Component",
    "Gender",
    "BMI_class",
    "Height_class",
}


def _values_from_height(frame: pd.DataFrame, height_m: float) -> dict[str, float]:
    """Resolve every numeric measurement directly from an actual stature.

    Each measurement is regressed against the real, per-subject stature using
    a monotone PCHIP curve fitted on stature-binned means, then evaluated at
    ``height_m``. This models each measurement's true correlation with
    stature, so the same curve interpolates within the ANSUR range and
    extrapolates beyond it, instead of relying on a percentile lookup that is
    undefined outside ``[0, 100]``.
    """
    stature = frame["stature_m"].to_numpy(dtype=float)
    n_bins = int(np.clip(len(frame) // 100, 10, 40))
    bins = pd.qcut(stature, n_bins, duplicates="drop")

    values: dict[str, float] = {}
    for column in frame.columns:
        if column in _EXCLUDED_COLUMNS:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = numeric.notna().to_numpy()
        if not valid.any():
            continue
        means = (
            pd.DataFrame({"stature": stature[valid], "metric": numeric.to_numpy(dtype=float)[valid]})
            .groupby(bins[valid], observed=True)
            .mean()
            .sort_values("stature")
        )
        interpolator = _pchip_with_linear_tails(
            means["stature"].to_numpy(), means["metric"].to_numpy()
        )
        values[column] = interpolator(height_m)
    return values


def _percentile_values(frame: pd.DataFrame, percentile: float) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in frame.columns:
        if column in _EXCLUDED_COLUMNS:
            continue
        numeric = (
            pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)
        )
        if numeric.size == 0:
            continue
        values[column] = float(np.percentile(numeric, percentile, method="linear"))
    return values


def resolve_reference(
    gender: str,
    height: float | None = None,
    percentile: float = 50.0,
    dataset: str | Path = DEFAULT_DATASET,
) -> AnthropometricReference:
    """Resolve one anthropometric reference using a common ANSUR percentile.

    ``height`` is expressed in centimetres. When present, every numeric
    measurement is resolved directly from that stature (via a per-measurement
    PCHIP regression against ANSUR subjects), and its empirical percentile is
    reported for information only; both take precedence over ``percentile``.
    When absent, ``percentile`` is applied to every numeric ANSUR measurement,
    including stature, using plain NumPy quantiles. Heights outside the ANSUR
    stature range are extrapolated with PCHIP interpolators and emit a
    :class:`UserWarning`.

    Parameters
    ----------
    gender : str
        Sex code, either ``"M"`` or ``"F"``.
    height : float or None, optional
        Requested stature in centimetres. If provided, its empirical ANSUR
        percentile overrides ``percentile``.
    percentile : float, optional
        Common target percentile when ``height`` is not provided. Must be in
        the inclusive interval ``[0.1, 99.9]``.
    dataset : str or pathlib.Path, optional
        Path to the ANSUR CSV file.

    Returns
    -------
    AnthropometricReference
        Resolved measurements and the effective percentile.

    Raises
    ------
    ValueError
        If the gender, height, percentile, or dataset values are invalid.
    """
    normalized_gender = str(gender).strip().upper()
    if normalized_gender not in {"M", "F"}:
        raise ValueError("gender must be 'M' or 'F'")
    if height is not None and (not np.isfinite(height) or height <= 0):
        raise ValueError("height must be a positive finite value in centimetres")
    if not np.isfinite(percentile) or not 0.1 <= percentile <= 99.9:
        raise ValueError("percentile must be between 0.1 and 99.9")

    frame = load_ansur(dataset)
    gender_frame = frame.loc[frame["Gender"] == normalized_gender]
    if gender_frame.empty:
        raise ValueError(f"ANSUR has no records for gender {normalized_gender!r}")

    if height is not None:
        target_percentile = _height_percentile(
            gender_frame["stature_m"].to_numpy(dtype=float),
            float(height),
        )
        target_percentile = min(99.9, max(0.1, target_percentile))
        # Measurements come straight from the actual stature, not from
        # target_percentile: a percentile lookup is undefined outside
        # [0, 100] and would not reflect this exact height anyway.
        values = _values_from_height(gender_frame, float(height) / 100.0)
    else:
        target_percentile = percentile
        values = _percentile_values(gender_frame, target_percentile)

    return AnthropometricReference(
        normalized_gender,
        target_percentile,
        values,
    )
