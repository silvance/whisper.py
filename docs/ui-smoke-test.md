# Whispers UI smoke test

Automated checks cannot prove a graphical interface is right. `pytest`, `ruff`
and `mypy` say the code is sound; only a person at the window can say whether
the screen explains itself. This is the list to work through before a build
goes out, and after any change to `whispr/gui/`.

Run it on the deployment machine if you can — Windows scaling and the platform
font change spacing, and that is exactly what this list is for.

## Before anything else: is this copy intact?

Do this first, on the machine that will actually run it, and before reading any
traceback. A bundle damaged in transit fails at whichever file it reaches first,
which can be a different file on each run.

- [ ] From a terminal in the extracted folder, `whispr --verify`
      (`whispr.exe --verify` on Windows) reports that the copy matches the build
      and exits 0.
- [ ] If it names missing, altered or unreadable files, the copy is the problem
      and not the software. Extract the archive again with 7-Zip rather than
      Windows Explorer, which is unreliable on archives of several GB; copy the
      extracted folder to a local disk rather than running it from removable
      media; and check whether a scanner has quarantined anything.
- [ ] If it reports that there is no inventory to check against, the bundle
      predates this check - rebuild before deploying.
- [ ] `whispr --self-test` then prints the report and exits 0, naming the build
      ID and commit and listing the models actually bundled.

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
      subtitles, formatting and the operation profile. The number of speakers is
      *not* in there — a line in Speaker separation says where it went.
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
- [ ] **Skip silence** (Advanced options > Audio and output) carries a note
      saying it can still cut very quiet speech and can be turned off.
- [ ] On a recording with long gaps, the line above the transcript says what
      share of it was listened to; on a clean recording it says nothing.
- [ ] When silence skipping passes over most of a recording, an amber banner
      says how many minutes went unheard and how to turn it off — and the same
      line is in the Status tab.
- [ ] Turning **Skip silence** off transcribes the whole recording and the
      "% listened to" note disappears.

### How many people are speaking

- [ ] With **Identify who is speaking** ticked, the question sits directly under
      it in Options, reading "Choose…" on a first launch, with a note about what
      "Not sure" costs.
- [ ] Selecting **Transcribe recording** without answering does not start the
      run: the settings reopen, an amber banner asks the question, the line
      under the progress bar says what it is waiting for, and the dropdown takes
      focus.
- [ ] Answering it takes the banner down and returns the line to Idle; the same
      button then starts the run.
- [ ] Choosing a number shows that many **Speaker names** fields in Advanced
      options; "Not sure" shows none.
- [ ] Unticking **Identify who is speaking** removes the question entirely, and
      the run starts without it.
- [ ] The answer is remembered across a relaunch and travels with an operation
      profile. An operator upgrading from a build that never asked is asked once.

### Redoing the speaker split

- [ ] **Redo speaker separation…** sits under the export row. Before any run it
      is greyed out and says it becomes available after a run with speaker
      identification on.
- [ ] After a diarized run it is enabled and the line beside it names how many
      speakers were found.
- [ ] It is greyed out again while a run is in progress.
- [ ] The dialog names the recording, the answer the last run used and the count
      it came back with, and offers every count *except* "Choose…".
- [ ] With speaker tags corrected by hand, the dialog says in amber how many will
      be replaced; with none, it says only that the words stay as transcribed.
- [ ] **Cancel** changes nothing.
- [ ] **Redo separation** re-splits without transcribing again — it finishes in a
      fraction of the original run's time, the progress bar names the speaker
      step, and a green banner says how many speakers came back.
- [ ] The words are identical to before; only the speaker tags have changed. Any
      names given to the old speakers are gone.
- [ ] With an output folder in use, the saved .txt (and .srt) are rewritten to
      match what is on screen.
- [ ] **Analysis report…** afterwards names the count the redo used, not the
      original run's.
- [ ] After a plain transcribe-only run (speaker identification off), the button
      stays greyed out.
- [ ] Starting a new transcription greys it out again until that run finishes.

### The transcript in its own window

- [ ] **Open in its own window** sits at the top right of the Transcript card.
      Selecting it gives the transcript a real window — titled, resizable,
      maximisable, and movable to a second screen.
- [ ] Everything came with it: both tabs, Find, the export buttons and the redo
      button. The button now reads **Put it back on this page**.
- [ ] The page shows a stand-in card with **Bring it back here** and **Show me
      that window**; the second raises the window when it is behind something.
- [ ] A transcription run started from the main window still fills the popped-out
      transcript and Status tabs as it goes.
- [ ] Closing the window with its X puts the panel back on the page rather than
      destroying it.
- [ ] Round-trip it twice: the transcript, scroll position and speaker tags are
      unchanged, and clicking a speaker tag still reassigns speech.
- [ ] The same button is on the **Live** tab's transcript card, and a live feed
      keeps writing into it while it is popped out.
- [ ] On a smaller screen the window opens fully on-screen, with its buttons
      reachable.

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

## Comparison History

- [ ] Running a comparison on Compare Speakers adds a row here without a
      restart, and the row names the recording, the subject and the score.
- [ ] Every outcome is listed, not only the interesting ones: high,
      intermediate, low, insufficient data, and a refused model mismatch.
- [ ] **Type** distinguishes a 1:1 comparison from a gallery search.
- [ ] The **Subject** and **Recording** filters both narrow the list, and the
      line underneath says how many of how many are shown.
- [ ] Selecting a row fills the detail: score against the threshold in force at
      the time, the questioned speech and the ranges it came from, the source
      SHA-256, what the reference profile held **at the time**, the voice model
      and the application version.
- [ ] A gallery row shows the whole ranking, the margin over the runner-up, and
      no reference-profile facts (there was no single reference).
- [ ] The disclaimer sits at the end of the detail, below the ranking and any
      caveats — never above them.
- [ ] Deleting a speaker profile leaves its comparisons in the history, marked
      **(deleted)**.
- [ ] Moving or removing a questioned recording shows "not at this location
      now" against the record, which is otherwise unchanged.
- [ ] **Export as CSV…** writes the filtered rows; the file has one row per
      comparison and no percentages anywhere.
- [ ] **Delete entry** is disabled with nothing selected, looks destructive, and
      asks before removing one record.
- [ ] Nowhere on the page is a score shown as a percentage, a "match", or a
      confirmation of identity.

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
