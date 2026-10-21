"""GroupCycle3 exercise-bike equipment model."""

from .bike import Bike

__all__ = ["GroupCycle3"]


class GroupCycle3(Bike):
    """A wrapper of Bike to fit the GroupCycle3 specs."""

    def __init__(self):
        super().__init__(
            # piantone seduta
            seat_tube_range_mm=((-150, 800), (-226, 1111)),
            seat_tube_steps=23,
            seat_length_mm=260.0,
            seat_width_mm=140.0,
            # slitta sella
            seat_slide_range_mm=(0, 150),
            seat_slide_steps=25,
            seat_slide_angle_deg=0.0,
            # piantone manubrio
            handlebar_tube_range_mm=((430.0, 970.0), (411.0, 1078.0)),
            handlebar_tube_steps=10,
            # slitta manubrio
            handlebar_slide_range_mm=(0, 120),
            handlebar_slide_steps=10,
            handlebar_slide_angle_deg=10.0,
            # handlebar
            handlebar_width_mm=450,
            handlebar_diameter_mm=40.0,
            # horns
            horn_pitch_deg=33.0,
            horn_roll_deg=10.0,
            horn_length_mm=120.0,
            # screen parameters
            screen_width_mm=60.0,
            screen_height_mm=110.0,
            screen_inclination_deg=40.0,
            screen_vertical_offset_mm=32.0,
            screen_horizontal_offset_mm=217.0,
            # frame
            frame_tube_diameter_mm=40.0,
            frame_width_mm=500.0,
            frame_margin_mm=300.0,
            # crank and pedal parameters
            crank_axis_position_mm=(0.0, 322.0, 0.0),
            crank_spacing_mm=170.0,
            crank_length_mm=172.5,
            pedal_spacing_mm=185.0,
            pedal_size_mm=(110.0, 70.0, 25.0),
            seat_height_mm=50.0,
        )
