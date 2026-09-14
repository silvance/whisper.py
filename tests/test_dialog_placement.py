"""Where a question opens when the application is not on the primary screen.

The failure this holds shut was reported from the field: the main window on a
second monitor, a dialog asking which speech belongs to the subject, and the
dialog opening in the top-left corner of the *first* monitor - behind a
browser, unseen, while the application sat waiting on an answer and looked to
its operator like it had frozen.

Left to the window manager that is where a Tk dialog goes. So it is not left
to the window manager.
"""

from whispr.dialog_placement import Rect, geometry, place

# A main window on a second monitor, to the right of a 2560-wide primary one.
RIGHT = Rect(2560, 200, 1400, 900)
# And one to the left of it, where screen coordinates are negative.
LEFT = Rect(-1920, 0, 1400, 900)


def test_a_dialog_opens_centred_on_the_window_that_asked():
    x, y = place((400, 200), RIGHT)
    assert x == 2560 + (1400 - 400) // 2
    assert y == 200 + (900 - 200) // 2


def test_a_dialog_follows_the_window_to_the_second_monitor():
    """The regression: it used to open wherever the window manager chose."""
    x, _ = place((400, 200), RIGHT)
    assert x >= RIGHT.x
    assert x + 400 <= RIGHT.x + RIGHT.width


def test_a_monitor_to_the_left_keeps_its_negative_coordinates():
    """Clamping x to 0 - the obvious-looking guard - drags it back one screen."""
    x, y = place((400, 200), LEFT)
    assert x < 0
    assert LEFT.x <= x <= LEFT.x + LEFT.width - 400


def test_a_dialog_never_spills_past_the_window_it_belongs_to():
    for size in ((320, 180), (900, 600), (1399, 899)):
        x, y = place(size, RIGHT)
        assert RIGHT.x <= x and x + size[0] <= RIGHT.x + RIGHT.width
        assert RIGHT.y <= y and y + size[1] <= RIGHT.y + RIGHT.height


def test_a_dialog_too_big_for_the_window_starts_at_its_corner():
    """Centring an oversized dialog would hide its title bar and its close box."""
    assert place((2000, 1200), RIGHT) == (RIGHT.x, RIGHT.y)


def test_an_undrawn_window_is_not_guessed_at():
    """A window reports 1x1 before it is drawn; centring on that is the corner."""
    assert place((400, 200), Rect(0, 0, 1, 1)) is None
    assert not Rect(0, 0, 1, 1).usable


def test_a_real_window_is_usable():
    assert RIGHT.usable and LEFT.usable


def test_the_geometry_string_is_the_one_tk_understands():
    assert geometry((400, 200), (2960, 550)) == "400x200+2960+550"


def test_a_negative_position_is_written_so_tk_reads_it_as_a_position():
    """Bare "-40" means 40 from the opposite edge - the far side of the desktop."""
    assert geometry((400, 200), (-1660, 40)) == "400x200+-1660+40"


def test_a_zero_size_is_never_written():
    """Tk rejects a 0-wide geometry; a dialog with no content should still open."""
    assert geometry((0, 0), (10, 10)) == "1x1+10+10"
