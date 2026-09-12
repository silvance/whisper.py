"""Where a popped-out panel opens, and what its button says.

The window is only *placed* here — the whole point of the feature is that the
operator then moves and maximises it on whichever screen they like. What these
guard is that it never opens somewhere unusable.
"""

import pytest

from whispr.panel_window import (
    DOCK_LABEL,
    MIN_HEIGHT,
    MIN_WIDTH,
    POPOUT_LABEL,
    fit_geometry,
)


def _parse(geometry):
    size, x, y = geometry.replace("+", " ").split()[0], *geometry.split("+")[1:]
    width, height = (int(n) for n in size.split("x"))
    return width, height, int(x), int(y)


def test_a_big_screen_gets_the_size_asked_for():
    width, height, _, _ = _parse(fit_geometry("1100x780", 2560, 1440))
    assert (width, height) == (1100, 780)


def test_a_small_screen_cuts_the_window_down_to_fit():
    """A laptop panel after a desktop monitor: buttons must not fall off it."""
    width, height, x, y = _parse(fit_geometry("1100x780", 1366, 768))
    assert width <= 1366 and height <= 768
    assert x + width <= 1366
    assert y + height <= 768


def test_the_window_never_opens_off_the_top_or_left():
    for screen in ((640, 480), (800, 600), (1024, 768), (3840, 2160)):
        _, _, x, y = _parse(fit_geometry("1100x780", *screen))
        assert x >= 0 and y >= 0


def test_a_tiny_screen_still_gets_a_usable_window():
    """Clamping has a floor: a 2-pixel window would be worse than no feature."""
    width, height, _, _ = _parse(fit_geometry("1100x780", 320, 240))
    assert width >= MIN_WIDTH
    assert height >= MIN_HEIGHT


@pytest.mark.parametrize("size", ["1100x780", "720x480", "1600x900"])
def test_the_wanted_size_is_a_ceiling_not_a_floor(size):
    want_w, want_h = (int(n) for n in size.split("x"))
    width, height, _, _ = _parse(fit_geometry(size, 2560, 1440))
    assert width <= want_w and height <= want_h


def test_the_button_says_something_different_each_way_round():
    """One button, two jobs - it has to read as a different offer each time."""
    assert POPOUT_LABEL != DOCK_LABEL
    assert POPOUT_LABEL and DOCK_LABEL
