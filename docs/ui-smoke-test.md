# Whispers UI smoke test

Automated checks cannot prove a graphical interface is right. `pytest`, `ruff`
and `mypy` say the code is sound; only a person at the window can say whether
the screen explains itself. This is the list to work through before a build
goes out, and after any change to `whispr/gui/`.

Run it on the deployment machine if you can — Windows scaling and the platform
font change spacing, and that is exactly what this list is for.

## Startup

- [ ] The window opens at a sensible size and can be resized down to its
      minimum (940×600) without controls being clipped or overlapping.
- [ ] The header shows **Whispers**, "Offline audio analysis", the green
      **Local processing** marker, **Help** and **System status**.
- [ ] The navigation rail lists only the pages this build supports. On a
      transcribe-only build (or `WHISPR_MODE=transcribe`) the rail is absent
      and the Transcribe page fills the window.
- [ ] Nothing renders as a pale box on the dark ground — text areas,
      scrollbars, tables and dialogs all take the application's own colours.

## Navigation

- [ ] Each rail entry raises its page; the current page is the highlighted one.
- [ ] Switching pages and returning preserves what was on the first page
      (chosen file, transcript, comparison result).

## Transcribe

- [ ] **Empty:** the page title and sentence are visible, the Recording box
      invites a drop, and the transcript area says what to do next.
- [ ] **Choose file** opens the picker; the chosen recording shows its name and
      size, with the folder as quiet metadata, and a **Change file** action.
- [ ] **Drag and drop** a recording onto the Recording box, the transcript pane
      and the Status pane — each loads the file.
- [ ] The **Quality** dropdown reads "Fast — base.en" and similar; picking one
      keeps the availability note underneath correct.
- [ ] **Advanced options** opens and closes, and contains hardware, silence
      skipping, the speaker method, sensitivity, expected words, conversion,
      subtitles, formatting and the operation profile.
- [ ] **Running:** the primary button is disabled, Cancel is enabled, the
      progress bar moves, and the line beneath it names the current step in
      words. The settings put themselves away.
- [ ] **Complete:** a green banner says so, the transcript has the page, and
      the line above it names the recording, language, duration and speakers.
- [ ] **Cancelled** and **failed** runs both say so where the eye already is,
      and a failure names the friendly reason and points at the Status tab. No
      traceback appears outside the Status tab.
- [ ] **No file chosen:** selecting Transcribe recording says so rather than
      failing silently.
- [ ] Find, Copy transcript, Save as Word…, Analysis report…, Save project…,
      Open project… all still work, and clicking a speaker tag or dragging a
      run of words still reassigns speech.
- [ ] A batch of several files still runs, and the queue summary appears with a
      **Clear list** action only when there is a queue.

## Saving a corrected speaker to a subject

- [ ] After a diarized run, **Save speaker to profile…** sits under the export
      row with a one-line explanation beside it. It is absent in a build with no
      speaker-embedding model.
- [ ] Correct some speaker tags first; the dialog then lists the speakers under
      the names you gave them, longest first, with how much speech each has.
- [ ] The dialog offers an existing subject *or* a new one, and says that
      samples arrive pending review. Escape closes it; Enter confirms.
- [ ] Typing the name of a subject that already exists adds to that subject
      rather than creating a second one with the same name.
- [ ] After saving, the banner names how many samples were added and that they
      need approval, and Speaker Profiles shows them under **Needs review** —
      not counted in the trusted reference speech.
- [ ] Saving the same speaker twice from the same recording adds nothing the
      second time, and the Status tab explains why.
- [ ] With no transcript, no diarized speakers, or no audio left on hand, the
      button explains what is missing instead of failing silently.

## Speaker Profiles

- [ ] **Empty:** both halves explain themselves rather than showing empty boxes.
- [ ] **Populated:** the list shows who exists; a subject with samples awaiting
      review is marked in the list as well as in the detail panel.
- [ ] Selecting a subject fills the detail panel: reference speech, trusted
      samples, awaiting review, source recordings, voice model.
- [ ] **Pending samples** are visually distinct from trusted ones, and one that
      is unlike the rest of the reference is distinct again.
- [ ] **Approve sample** is disabled when nothing is pending.
- [ ] **Add a recording** offers whole file / diarized speaker / time ranges,
      and each still enrols.
- [ ] Importing a damaged profile shows the dialog **and** leaves a warning
      banner on the page after it is dismissed.
- [ ] **Delete profile** looks destructive and asks before deleting.

## Compare Speakers

- [ ] The three stages read in order: reference, questioned recording, result.
- [ ] With no reference selected, stage 1 says where to create one.
- [ ] **High similarity:** the band is the largest thing on the result, the
      sentence reads as a lead, and the score reads `0.78 / 1.00`.
- [ ] **Low similarity** and **no sufficiently strong match** read plainly.
- [ ] **Insufficient data** visually outranks the score: the band says so and
      the sentence says the score is not meaningful, even when the number is
      high.
- [ ] **Refused** (a profile from another voice model) says so in the same
      place and gives the reason.
- [ ] **Search all profiles** on poor audio shows the ranking but states that
      it supports no conclusion.
- [ ] Nowhere does the screen show a percentage, "match", "confirmed" or
      "same person". The disclaimer is visible without expanding anything.
- [ ] **Full result text** expands to the copyable block; Export report… and
      Copy result work.

## Live and Translate

- [ ] Each opens with its title and sentence, and has one obvious primary
      action.
- [ ] Live: Start / Stop / Test connection behave as before, and the transcript
      area fills as text arrives.
- [ ] Translate: the paste box, Extract from image/PDF…, and the file batch all
      still work.

## Dialogs

- [ ] Help and System status open themed, scroll, close on **Escape**, and open
      with the Close button focused.
- [ ] System status still leads with READY / NOT READY.
- [ ] Compare voices… (from the operation profile card) is themed and readable.

## Scrolling

- [ ] Every page scrolls with the wheel and with the scrollbar, at 1366×768 and
      when the window is made deliberately small.
- [ ] **Hide settings** / **Show settings** on Transcribe, and expanding or
      collapsing **Advanced options**, leave the page still scrollable — with no
      resize, maximise or page switch in between.
- [ ] Starting a transcription (which puts the settings away on its own) leaves
      the page scrollable, and **Show settings** during a run does too.
- [ ] Scrolling to the bottom and then hiding the settings does not leave the
      page parked on blank space below its own content.
- [ ] The wheel scrolls over controls that appeared after the page was built
      (a result card, an expanded section), and one notch still moves one notch.
- [ ] The wheel over the transcript and Status panes scrolls those panes, not
      the page behind them.

## Layout and accessibility

- [ ] **1366×768:** every page is usable; the primary action is reachable
      without hunting, and anything below the fold scrolls with the wheel and
      the scrollbar.
- [ ] **1920×1080:** cards and columns fill the width without stranding
      controls at one edge.
- [ ] **Windows display scaling at 125%:** text is not clipped and controls do
      not overlap.
- [ ] **Tab** moves through the controls in a sensible order and the focused
      control is visibly focused.
- [ ] No state is carried by colour alone: pending samples, warnings and errors
      all say what they are in words.
- [ ] **Stock Tk** (uninstall or hide ttkbootstrap): the application still
      starts and every page is usable, if plainer.
