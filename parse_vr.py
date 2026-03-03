"""Parse raw VR controller string data from Unity into structured dicts."""

from __future__ import annotations

import re
import utils

EXPECTED_FIELD_COUNT = 14


def parse_data(input_data: str | None) -> list[dict] | None:
    """Parse controller CSV string into a list of controller dicts.

    Each controller block starts with 'LTouch' or 'RTouch' followed by
    14 numeric values (3 pos + 4 rot + 2 joystick + 1 index + 1 grip +
    1 AX + 1 BY + 1 joystick_press).
    """
    if not input_data:
        return None

    input_data = input_data.strip()
    chunks = re.split(r"(?=(?:LTouch|RTouch),)", input_data)

    controllers = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        parts = chunk.split(",")
        name = parts[0]

        try:
            nums = [float(p) for p in parts[1:] if p]
        except ValueError:
            utils.logger.warning(f"Non-numeric value in VR data chunk: {chunk}")
            continue

        if len(nums) != EXPECTED_FIELD_COUNT:
            utils.logger.warning(
                f"Unexpected field count for {name}: {len(nums)} "
                f"(expected {EXPECTED_FIELD_COUNT})"
            )
            continue

        controllers.append({
            "ControllerType": name,
            "Position": tuple(nums[0:3]),
            "Rotation": tuple(nums[3:7]),
            "Joystick": tuple(nums[7:9]),
            "IndexTrigger": nums[9],
            "GripTrigger": nums[10],
            "Button_AX": int(nums[11]),
            "Button_BY": int(nums[12]),
            "Joystick_Press": int(nums[13]),
        })

    return controllers
