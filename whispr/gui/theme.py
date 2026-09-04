"""The one place the application's visual language is defined.

Whispers is used by people who are not developers, often occasionally, on an
air-gapped machine where there is nobody to ask. The interface therefore has to
carry its own explanation: a clear hierarchy, one obvious action per screen, and
technical depth available but out of the way.

This module holds the tokens that produce that consistency - spacing, colour,
type roles - and registers the ttk styles built from them, so a screen asks for
``Style.PAGE_TITLE`` rather than inventing another font tuple. Nothing here
knows anything about transcription; it is presentation only.

Everything is local: system fonts, no downloaded assets, no network. Under
ttkbootstrap the styles ride on its themed widgets; under stock Tk they degrade
to something plainer but still usable, which is the deal the fallback has always
made.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import font as tkfont
from tkinter import ttk
from typing import Optional

# -- Spacing ---------------------------------------------------------------
# One scale, used everywhere. Mixed ad-hoc padding is what makes an interface
# look assembled rather than designed.

SPACE_XS = 4  # inside a control group
SPACE_SM = 8  # between related controls
SPACE_MD = 12  # normal control spacing
SPACE_LG = 16  # generous control spacing
SPACE_XL = 24  # between sections
SPACE_XXL = 32  # between major page regions

# Card internals: enough room that content is not pressed against the edge.
CARD_PADDING = (SPACE_LG, SPACE_MD, SPACE_LG, SPACE_MD)
PAGE_PADDING = (SPACE_XL, SPACE_MD, SPACE_XL, SPACE_XL)


@dataclass(frozen=True)
class Palette:
    """Colour by meaning, not decoration.

    Dark by default: this is a long-session analysis tool, often in a room where
    a bright window is unwelcome. One accent carries "the action to take"; the
    status colours are reserved for status, so that when something does turn
    amber or red it reads as information rather than styling.
    """

    background: str = "#1b1d21"  # application ground
    surface: str = "#24272c"  # cards sitting on it
    surface_alt: str = "#2b2f35"  # nested surfaces, hover, selected rows
    border: str = "#343941"  # hairlines, used sparingly
    text: str = "#e9ebee"
    text_muted: str = "#9aa2ad"  # explanatory copy
    text_faint: str = "#6e7681"  # metadata, timestamps, hashes
    accent: str = "#2f6feb"  # the primary action
    accent_active: str = "#4884f2"
    accent_text: str = "#ffffff"
    success: str = "#2ea043"
    warning: str = "#d29922"
    danger: str = "#c9403f"
    danger_active: str = "#d9534f"
    # Tinted grounds for status banners - readable without shouting.
    success_bg: str = "#17301d"
    warning_bg: str = "#332813"
    danger_bg: str = "#331c1c"
    info_bg: str = "#17263f"


class Style:
    """ttk style names, so screens never spell them as strings."""

    # Surfaces
    PAGE = "Page.TFrame"
    CARD = "Card.TFrame"
    CARD_INNER = "CardInner.TFrame"
    HEADER = "Header.TFrame"
    SIDEBAR = "Sidebar.TFrame"
    SEPARATOR = "Thin.TSeparator"

    # Type roles
    APP_TITLE = "AppTitle.TLabel"
    APP_SUBTITLE = "AppSubtitle.TLabel"
    PAGE_TITLE = "PageTitle.TLabel"
    PAGE_SUBTITLE = "PageSubtitle.TLabel"
    SECTION_TITLE = "SectionTitle.TLabel"
    BODY = "Body.TLabel"
    MUTED = "Muted.TLabel"
    META = "Meta.TLabel"
    FIELD_LABEL = "FieldLabel.TLabel"
    MONO = "Mono.TLabel"

    # Status text
    SUCCESS = "Success.TLabel"
    WARNING = "Warning.TLabel"
    DANGER = "Danger.TLabel"
    ACCENT = "Accent.TLabel"

    # Buttons
    PRIMARY = "Primary.TButton"
    SECONDARY = "Secondary.TButton"
    SUBTLE = "Subtle.TButton"  # low-emphasis, e.g. header actions
    DANGER_BUTTON = "Danger.TButton"
    NAV = "Nav.TButton"
    NAV_ACTIVE = "NavActive.TButton"
    LINK = "Link.TButton"  # "Advanced options" disclosure

    # Other
    CHECK = "Surface.TCheckbutton"
    RADIO = "Surface.TRadiobutton"


@dataclass
class Theme:
    """The initialised theme: the palette plus the fonts built from it."""

    palette: Palette
    body: tkfont.Font
    muted: tkfont.Font
    meta: tkfont.Font
    section: tkfont.Font
    page_title: tkfont.Font
    app_title: tkfont.Font
    mono: tkfont.Font
    bootstrap: bool = False

    @property
    def colors(self) -> Palette:
        return self.palette


_theme: Optional[Theme] = None


def theme() -> Theme:
    """The active theme. :func:`init_theme` must have run first."""
    if _theme is None:  # pragma: no cover - a programming error, not a state
        raise RuntimeError("init_theme() has not been called")
    return _theme


def palette() -> Palette:
    """Shorthand for the active colours."""
    return theme().palette


def _base_family(root: tk.Misc) -> str:
    """The platform's own UI font - Segoe UI on Windows, and so on.

    System fonts only: a bundled font file is one more thing to ship, licence
    and get wrong on an air-gapped machine, for no benefit an operator would
    notice.
    """
    try:
        return str(tkfont.nametofont("TkDefaultFont").actual("family"))
    except Exception:  # noqa: BLE001 - fall back to Tk's own default name
        return "TkDefaultFont"


def _mono_family(root: tk.Misc) -> str:
    try:
        return str(tkfont.nametofont("TkFixedFont").actual("family"))
    except Exception:  # noqa: BLE001
        return "TkFixedFont"


def init_theme(root: tk.Misc, *, bootstrap: bool = False) -> Theme:
    """Build the fonts, register the styles, and return the active theme.

    Called once, from the application shell, before any screen is built.
    """
    global _theme
    colors = Palette()
    family = _base_family(root)
    mono_family = _mono_family(root)

    # A restrained ramp. Type does the hierarchy; weight is used sparingly, so
    # that bold still means something where it appears.
    built = Theme(
        palette=colors,
        body=tkfont.Font(family=family, size=10),
        muted=tkfont.Font(family=family, size=9),
        meta=tkfont.Font(family=family, size=8),
        section=tkfont.Font(family=family, size=11, weight="bold"),
        page_title=tkfont.Font(family=family, size=17, weight="bold"),
        app_title=tkfont.Font(family=family, size=14, weight="bold"),
        mono=tkfont.Font(family=mono_family, size=9),
        bootstrap=bootstrap,
    )
    _theme = built

    style = ttk.Style(root)
    _configure_base(style, built)
    _configure_surfaces(style, built)
    _configure_text(style, built)
    _configure_buttons(style, built)
    _configure_inputs(style, built)
    try:
        root.configure(background=colors.background)  # type: ignore[call-arg]
    except tk.TclError:  # pragma: no cover - some widgets reject it
        pass
    return built


def _configure_base(style: ttk.Style, t: Theme) -> None:
    c = t.palette
    style.configure(".", background=c.background, foreground=c.text, font=t.body)
    style.configure("TFrame", background=c.background)
    style.configure("TLabel", background=c.background, foreground=c.text, font=t.body)
    for name in ("TCheckbutton", "TRadiobutton"):
        _safe_configure(
            style,
            name,
            background=c.background,
            foreground=c.text,
            indicatorbackground=c.surface_alt,
            indicatorforeground=c.accent,
            indicatorcolor=c.surface_alt,
        )
    style.configure(Style.SEPARATOR, background=c.border)


def _safe_configure(style: ttk.Style, name: str, **options: object) -> None:
    """Apply the options a theme understands and ignore the ones it does not.

    ttk themes disagree about which element options exist - a checkbutton
    indicator is spelled differently under clam, ttkbootstrap and the native
    Windows themes - and an unknown option raises rather than being ignored.
    Applying them one at a time keeps the styling that does apply.
    """
    for option, value in options.items():
        try:
            style.configure(name, **{option: value})
        except tk.TclError:  # pragma: no cover - theme-dependent
            continue


def _configure_surfaces(style: ttk.Style, t: Theme) -> None:
    c = t.palette
    style.configure(Style.PAGE, background=c.background)
    # Cards are separated by ground and space rather than by heavy borders: a
    # page of outlined boxes reads as a form, not as a product.
    style.configure(Style.CARD, background=c.surface, relief="flat", borderwidth=0)
    style.configure(Style.CARD_INNER, background=c.surface)
    style.configure(Style.HEADER, background=c.surface)
    style.configure(Style.SIDEBAR, background=c.surface)
    for name in (Style.CHECK, Style.RADIO):
        _safe_configure(
            style,
            name,
            background=c.surface,
            foreground=c.text,
            indicatorbackground=c.surface_alt,
            indicatorforeground=c.accent,
            indicatorcolor=c.surface_alt,
            focuscolor=c.accent,
            padding=(0, SPACE_XS),
        )
        style.map(
            name,
            background=[("active", c.surface)],
            indicatorbackground=[
                ("selected", c.accent),
                ("active", c.border),
            ],
            indicatorcolor=[("selected", c.accent), ("active", c.border)],
        )


def _configure_text(style: ttk.Style, t: Theme) -> None:
    c = t.palette
    roles = (
        (Style.APP_TITLE, t.app_title, c.text, c.surface),
        (Style.APP_SUBTITLE, t.muted, c.text_muted, c.surface),
        (Style.PAGE_TITLE, t.page_title, c.text, c.background),
        (Style.PAGE_SUBTITLE, t.muted, c.text_muted, c.background),
        (Style.SECTION_TITLE, t.section, c.text, c.surface),
        (Style.BODY, t.body, c.text, c.surface),
        (Style.MUTED, t.muted, c.text_muted, c.surface),
        (Style.META, t.meta, c.text_faint, c.surface),
        (Style.FIELD_LABEL, t.body, c.text_muted, c.surface),
        (Style.MONO, t.mono, c.text, c.surface),
        (Style.SUCCESS, t.body, c.success, c.surface),
        (Style.WARNING, t.body, c.warning, c.surface),
        (Style.DANGER, t.body, c.danger_active, c.surface),
        (Style.ACCENT, t.body, c.accent_active, c.surface),
    )
    for name, font, foreground, background in roles:
        style.configure(name, font=font, foreground=foreground, background=background)


def _configure_buttons(style: ttk.Style, t: Theme) -> None:
    c = t.palette
    # One filled accent button per workflow; everything else is quieter, so the
    # eye lands on the thing to do rather than on whichever button is nearest.
    style.configure(
        Style.PRIMARY,
        background=c.accent,
        foreground=c.accent_text,
        font=t.body,
        borderwidth=0,
        focusthickness=1,
        focuscolor=c.accent_text,
        padding=(SPACE_LG, SPACE_SM),
        anchor="center",
    )
    style.map(
        Style.PRIMARY,
        background=[("disabled", c.surface_alt), ("active", c.accent_active)],
        foreground=[("disabled", c.text_faint)],
    )
    style.configure(
        Style.SECONDARY,
        background=c.surface_alt,
        foreground=c.text,
        font=t.body,
        borderwidth=0,
        focusthickness=1,
        focuscolor=c.accent,
        padding=(SPACE_MD, SPACE_SM),
        anchor="center",
    )
    style.map(
        Style.SECONDARY,
        background=[("disabled", c.surface), ("active", c.border)],
        foreground=[("disabled", c.text_faint)],
    )
    style.configure(
        Style.SUBTLE,
        background=c.surface,
        foreground=c.text_muted,
        font=t.muted,
        borderwidth=0,
        focusthickness=1,
        focuscolor=c.accent,
        padding=(SPACE_MD, SPACE_XS),
    )
    style.map(
        Style.SUBTLE,
        background=[("active", c.surface_alt)],
        foreground=[("active", c.text)],
    )
    # Destructive actions must not look like ordinary ones.
    style.configure(
        Style.DANGER_BUTTON,
        background=c.surface_alt,
        foreground=c.danger_active,
        font=t.body,
        borderwidth=0,
        padding=(SPACE_MD, SPACE_SM),
    )
    style.map(
        Style.DANGER_BUTTON,
        background=[("active", c.danger), ("disabled", c.surface)],
        foreground=[("active", c.accent_text), ("disabled", c.text_faint)],
    )
    style.configure(
        Style.NAV,
        background=c.surface,
        foreground=c.text_muted,
        font=t.body,
        borderwidth=0,
        anchor="w",
        padding=(SPACE_MD, SPACE_SM),
    )
    style.map(
        Style.NAV,
        background=[("active", c.surface_alt)],
        foreground=[("active", c.text)],
    )
    style.configure(
        Style.NAV_ACTIVE,
        background=c.surface_alt,
        foreground=c.text,
        font=t.body,
        borderwidth=0,
        anchor="w",
        padding=(SPACE_MD, SPACE_SM),
    )
    style.map(Style.NAV_ACTIVE, background=[("active", c.surface_alt)])
    style.configure(
        Style.LINK,
        background=c.background,
        foreground=c.text_muted,
        font=t.muted,
        borderwidth=0,
        padding=(0, SPACE_XS),
        anchor="w",
    )
    style.map(
        Style.LINK,
        background=[("active", c.background)],
        foreground=[("active", c.text)],
    )


def _configure_inputs(style: ttk.Style, t: Theme) -> None:
    c = t.palette
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(
            name,
            fieldbackground=c.surface_alt,
            background=c.surface_alt,
            foreground=c.text,
            bordercolor=c.border,
            lightcolor=c.border,
            darkcolor=c.border,
            insertcolor=c.text,
            arrowcolor=c.text_muted,
            padding=(SPACE_SM, SPACE_XS),
        )
        style.map(
            name,
            bordercolor=[("focus", c.accent)],
            lightcolor=[("focus", c.accent)],
            darkcolor=[("focus", c.accent)],
            foreground=[("disabled", c.text_faint)],
        )
    _safe_configure(
        style,
        "Treeview",
        background=c.surface_alt,
        fieldbackground=c.surface_alt,
        foreground=c.text,
        bordercolor=c.surface_alt,
        lightcolor=c.surface_alt,
        darkcolor=c.surface_alt,
        borderwidth=0,
        relief="flat",
        rowheight=26,
    )
    style.map(
        "Treeview",
        background=[("selected", c.accent)],
        foreground=[("selected", c.accent_text)],
    )
    style.configure(
        "Treeview.Heading",
        background=c.surface,
        foreground=c.text_muted,
        font=t.muted,
        borderwidth=0,
        padding=(SPACE_SM, SPACE_XS),
    )
    style.map("Treeview.Heading", background=[("active", c.surface_alt)])
    # Sub-tabs (Transcript / Status): quiet until selected.
    _safe_configure(
        style,
        "TNotebook",
        background=c.surface,
        bordercolor=c.surface,
        borderwidth=0,
        tabmargins=(0, 0, 0, 0),
    )
    _safe_configure(
        style,
        "TNotebook.Tab",
        background=c.surface,
        foreground=c.text_muted,
        bordercolor=c.surface,
        borderwidth=0,
        padding=(SPACE_MD, SPACE_SM),
        font=t.body,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", c.surface_alt), ("active", c.surface_alt)],
        foreground=[("selected", c.text), ("active", c.text)],
    )
    style.configure(
        "TProgressbar",
        background=c.accent,
        troughcolor=c.surface_alt,
        bordercolor=c.surface_alt,
        lightcolor=c.accent,
        darkcolor=c.accent,
    )


def text_widget_options() -> dict:
    """Colours for a raw ``tk.Text``/``ScrolledText``, which ttk cannot style."""
    c = palette()
    return {
        "background": c.surface_alt,
        "foreground": c.text,
        "insertbackground": c.text,
        "selectbackground": c.accent,
        "selectforeground": c.accent_text,
        "highlightthickness": 0,
        "borderwidth": 0,
        "relief": "flat",
        "padx": SPACE_MD,
        "pady": SPACE_SM,
    }
