import math


def schedule_values(preset="reference_3", steps=3, start_sigma=0.909375, shift=6.0, custom=""):
    if preset == "reference_3":
        return [0.909375, 6.0 / 7.0, 2.0 / 3.0, 0.0]
    if preset == "reference_2":
        start_sigma, steps, shift = 0.7, 2, 6.0
    elif preset == "custom":
        values = [float(v) for v in custom.replace(",", " ").replace(";", " ").split()]
        validate_values(values)
        return values
    if steps < 1 or shift <= 0 or not 0 < start_sigma <= 1:
        raise ValueError("steps 必须为正数，shift 必须大于零，start_sigma 必须在 (0,1] 内。")
    raw_start = start_sigma / (shift - (shift - 1) * start_sigma)
    values = [start_sigma]
    for i in range(1, steps):
        raw = raw_start * (steps - i) / steps
        values.append(shift * raw / (1.0 + (shift - 1.0) * raw))
    return values + [0.0]


def validate_values(values):
    if len(values) < 2 or any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
        raise ValueError("SIGMAS 至少包含两个有限数值，范围为 [0,1]。")
    if values[-1] != 0 or any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Refiner SIGMAS 应严格递减并以 0 结束。")
