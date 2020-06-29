def tap_variation_limit(half_tap_range: int = 16, max_tap_deviation: float = 0.1, limit=0.02 ):
    width = max_tap_deviation / half_tap_range
    limits = {}

    for i in range(-half_tap_range, half_tap_range + 1):
        current_ratio = 1 + i * width
        max_deviation_ratio = (1 + limit) * current_ratio
        min_deviation_ratio = (1 - limit) * current_ratio

        to_tap = lambda x: (x - 1) / width
        limits[i] = [to_tap(min_deviation_ratio), to_tap(max_deviation_ratio)]

if __name__ == "__main__":
    tap_variation_limit()
