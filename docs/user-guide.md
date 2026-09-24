# Whispers — user guide

Whispers turns recordings into text you can read, search and quote, works out
who was speaking, and measures how similar a voice is to someone you already
have a reference for.

It runs entirely on the machine in front of you. Nothing is uploaded, nothing
is sent anywhere, and it does not need — or use — a network connection. The
models it uses are inside the bundle.

This guide is for the person using the tool. It assumes no technical
background and follows the order you would actually work in.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Transcribing a recording](#2-transcribing-a-recording)
3. [Working out who is speaking](#3-working-out-who-is-speaking)
4. [Fixing the transcript](#4-fixing-the-transcript)
5. [Getting the transcript out](#5-getting-the-transcript-out)
6. [A whole folder at once](#6-a-whole-folder-at-once)
7. [Reference voices: Speaker Profiles](#7-reference-voices-speaker-profiles)
8. [Comparing a voice: Compare Speakers](#8-comparing-a-voice-compare-speakers)
9. [Comparison History](#9-comparison-history)
10. [Live and Translate](#10-live-and-translate)
11. [When something goes wrong](#11-when-something-goes-wrong)
12. [What the results do and do not mean](#12-what-the-results-do-and-do-not-mean)

---

## 1. Before you start

### Checking the copy you have

Do this once, on the machine that will run it, before anything else. A bundle
damaged in transit fails in confusing ways, and it is worth ruling out first.

Open a terminal in the extracted folder and run:

```
whispr.exe --verify
```

It should report that the copy matches the build. Then:

```
whispr.exe --self-test
```

This prints what this copy can actually do. Read two lines in particular:

- **READY** / **NOT READY** at the top. NOT READY names what is missing.
- **Diarization: pyannote model cache** should say `bundled and readable`.
  Anything else means speaker separation will fail, and the build needs
  redoing — nothing you can change on this machine will fix it.

You can see the same report any time from **System status** in the top-right
of the window.

### The window

Down the left is a list of pages. Which ones appear depends on what this build
includes:

| Page | What it is for |
| --- | --- |
| **Transcribe** | Turning recordings into text. The main page. |
| **Speaker Profiles** | Building a reference voice for someone you know. |
| **Compare Speakers** | Measuring a voice against one of those references. |
| **Comparison History** | Every comparison this copy has run. |
| **Live** | Transcribing an incoming stream as it happens. |
| **Translate** | Translating text, and pulling text out of images and PDFs. |

The green **Local processing** marker in the header is there to be pointed at:
it means what it says.

---

## 2. Transcribing a recording

### Point it at a recording

Drag an audio or video file onto the window, or use **Choose file**. Audio and
video both work — video has its audio extracted first, automatically.

### Set the options

Everything most people need is in the **Options** card:

- **Language** — leave on **Auto** unless it guesses wrong, which happens on
  very short or very noisy recordings.
- **Quality** — how hard the tool works at hearing. The dropdown says what
  each choice means and names the model behind it: *Fast — base.en*,
  *Balanced — small.en*, *More thorough — medium.en*, *Thorough and quick —
  turbo*, *Most thorough, slowest — large-v3*. Slower means better on hard
  audio, and the difference on a poor phone line is large. Start at *Fast* for
  English and step up when the result is not good enough.

  The `.en` models are English only. Non-English audio needs one of the plain
  names (`small`, `medium`, `large-v3`, `turbo`). A note under the box says
  whether your pick is actually in this build.
- **Identify who is speaking** — tick this to label who said what. See the
  next section.
- **Save a copy to a folder** — where to write the `.txt` (and `.srt`) output.

Then press **Transcribe recording**.

Everything else lives under **Advanced options**, which you can ignore until
you need it. What is in there:

| Card | Holds |
| --- | --- |
| **Audio and output** | Skip silence, low-confidence highlighting, `.srt` subtitles, video conversion |
| **Language and vocabulary** | **Task** (*transcribe* keeps the language, *translate* produces English) and **Expected words** |
| **Speaker separation** | Method and grouping sensitivity — used only when *Identify who is speaking* is on |
| **Processing hardware** | CPU or GPU. *Auto* is right unless you are testing something |
| **Operation profile** | Settings and learned voices kept per operation — see below |

The text appears as it is produced. **Status** next to the Transcript tab
shows progress and anything worth telling you. The window stays usable
throughout, and **Cancel** stops the run.

### Worth knowing

- **Expected words** (*Advanced options* → *Language and vocabulary*) is the
  cheapest accuracy win available. Type in the names, places, callsigns and
  jargon likely to come up. It steers the model towards them, and proper nouns
  are what it otherwise mangles most.
- **Highlight low-confidence words** (*Advanced options* → *Audio and output*)
  colours the parts the model was unsure
  about, so you can go straight to what needs checking rather than re-reading
  the lot.
- **Ctrl-click any line or word** to hear just that span of the recording. Use
  it constantly. A transcript you have spot-checked against the audio is worth
  a great deal more than one you have not.
- **Open in its own window**, on the transcript card, puts the transcript on a
  second screen at full size while the settings stay where they are. **Put it
  back on this page** returns it.
- **Hide settings** gives the transcript the whole page.

### About Skip silence

**Skip silence** (*Advanced options* → *Audio and output*) passes over silent
stretches so a long recording with little talking finishes sooner.

On a quiet or surreptitious recording it can cut speech that is only slightly
above the noise. If it removes a lot, the banner at the end of the run says so
and names the recording. If speech seems to be missing, **turn Skip silence
off and run it again** — that is the first thing to try, before changing
anything else.

---

## 3. Working out who is speaking

Tick **Identify who is speaking** and the tool answers **How many people are
speaking?** before it will start. It has to be answered — it is the single
most useful thing you can tell it, and getting it wrong is expensive.

- **If you know the number, give it.** A two-person phone call transcribed as
  five speakers is the most common disappointment with this tool, and saying
  "2 people" prevents it outright.
- **Not sure — work it out** lets the tool decide. Use it when you genuinely do
  not know, and expect to check the result.

When you give a count, **Speaker names (optional)** fields appear. Fill them in
and the transcript comes out with real names rather than `SPEAKER_00`.

### If the split comes out wrong

Use **Redo speaker separation…** under the transcript. It splits the existing
transcript again with a different number — **without transcribing it all over
again**, which on a long recording is the difference between a minute and an
hour.

The words stay exactly as transcribed. Speakers are numbered from scratch, so
any names you typed have to be put back afterwards.

### Method and sensitivity

In *Advanced options* → *Speaker separation*, **Method** picks the engine.
Leave it on **Auto** unless you have a reason. In a bundle with both:

- **pyannote** is much better on hard, noisy or overlapping audio.
- **sherpa** is lighter and faster, and fine on clean audio.

**Grouping sensitivity** only applies to sherpa, and only when you have not
given a count. Higher merges more (fewer speakers), lower splits more.

### Operation profiles

*Advanced options* → **Operation profile** keeps your settings, and the voices
you have corrected, per operation rather than globally. Pick the profile you
are working under and the tool remembers what that work looks like — model,
options, and the voices it has learned from your corrections.

With **Recognise and learn speaker voices for this profile** on, a voice it
already knows from that operation is named in new transcripts automatically,
and voices you correct are learned for next time. Those learned samples still
wait for your approval before they count as a reference — see
[section 7](#7-reference-voices-speaker-profiles).

Leave it alone if you only work one case at a time. It earns its place when
you are moving between several.

---

## 4. Fixing the transcript

Speaker separation on difficult audio is never perfect. The transcript is
editable, and correcting it is expected work rather than a sign something went
wrong.

| To do this | Do that |
| --- | --- |
| Rename a speaker everywhere | Click their `[speaker]` tag |
| Move one line to another speaker | Click the tag, choose the individual option |
| Move a single word | Click that word |
| Move part of a line | Highlight the words and drag them onto the other speaker's line |

The last one is for when one person's sentence has been lumped into the middle
of another's. Right-click offers the same moves as a menu.

Which voice a diarizer calls `SPEAKER_00` is arbitrary, so if two names have
landed on the wrong people it is one click to swap them.

**Find**, under the transcript, jumps through matches — Enter for the next,
Shift+Enter for the previous.

---

## 5. Getting the transcript out

Under the transcript:

- **Copy transcript** — to the clipboard.
- **Save as Word…** — a `.docx`.
- **Analysis report…** — the transcript plus what produced it: which models,
  which settings, and a fingerprint of the source file. This is the one to
  export when the result may be questioned later.
- **Save project…** — writes everything, including your speaker names and
  corrections, to a `.whispr.json` file. **Open project…** brings it back so
  you can carry on in a later session.

Save a project whenever you have put real work into corrections. Re-doing them
is the most annoying way to lose an hour.

---

## 6. A whole folder at once

Drop a folder onto the window, or use **Add a folder…**. Every recording in it
is queued, including those in its subfolders. **Add several files…** does the
same for a hand-picked set.

The queue tells you what it took and what it left:

> 4 recordings queued from 3 folders — carpark.m4a, handover.mp3,
> interview.mp4, phone call.wav. 3 other files ignored; 1 already-converted
> copy left out (interview.wav).

Notes, photographs and last run's transcripts are ignored. A WAV that looks
like the converted copy a previous run made from a video of the same name is
left out too, so running the same folder twice does not transcribe the same
hour again. If one of those is genuinely a separate recording, add it with
**Choose file** and it will be used.

**Transcribe recording** then works through the queue. Each file's output is
written to the output folder, or beside the source if you have not set one,
and each transcript appears as it finishes.

**A recording it cannot read does not stop the run.** The failure is recorded,
the rest are transcribed, and the banner at the end says how many of how many
succeeded and what happened to the others. The detail is in the Status tab.
This matters when you leave a folder running overnight.

---

## 7. Reference voices: Speaker Profiles

A **profile** is a reference voice for a person you can already identify —
built from recordings you already have and are confident about. It is what a
later comparison is measured against, so what goes into it decides what every
comparison is worth.

### Building one

1. **New profile**, and name the subject.
2. **Add a recording** — a historical recording containing them.
3. Answer **Which speech is the subject?**
   - *The whole recording is this subject* — for a recording of them alone.
   - *Several people — separate speakers and let me pick* — for a conversation.
     You then pick which speaker is them from a list showing how long each
     talked.
   - *I'll give the time ranges* — type the spans yourself, e.g.
     `0:10-0:45, 1:20-2:00`.
4. Repeat with more recordings. More speech, from more occasions, in more
   conditions, makes a better reference.

**Reference samples** lists what the profile holds, each with its quality and
where it came from. The summary above it says how much **Reference speech**
the profile has, how many **Trusted samples**, and which **Voice model**
measured them. **Remove sample** takes out anything you are not happy with.

The voice model matters. A profile made by a different model is **always
refused** for comparison rather than measured anyway — the number would be
meaningless. (The *Allow a profile whose voice model cannot be verified*
option in Compare Speakers is only for an older profile that never recorded
which model made it, not a way around that refusal.)

### Samples awaiting review

When you correct a speaker in a transcript, the tool can learn that voice — but
it never folds it into a trusted reference on its own. Those samples wait under
**Awaiting review** until you press **Approve sample**. Listen before you do.
A sample approved carelessly quietly degrades every comparison that follows,
and nothing later will point back at it as the reason.

### For a conversation, the easier route

Transcribe it first with **Identify who is speaking** on, correct the speaker
tags there, then use **Save speaker to profile…** under the transcript. It is
usually less work than picking spans by hand.

### Moving profiles between machines

**Export…** writes a profile to a file; **Import…** reads one in. Both are
offline. A profile contains voice measurements and the case information you
typed — treat the file with the same care as the recordings it came from.

---

## 8. Comparing a voice: Compare Speakers

This measures how similar a speaker in a questioned recording is to a
reference profile. The page is three numbered steps, in order:

1. **Reference speaker** — pick the subject whose profile you are measuring
   against.
2. **Questioned recording** — choose the recording, then answer **Whose voice
   should be measured?**: the whole recording, or one speaker in it, which you
   pick from a list showing how long each talked.
3. **Result** — what was found.

### Reading the result

The result is a **similarity band**, not a verdict:

| Band | What it means |
| --- | --- |
| **High similarity** | The questioned speech is highly similar to the reference voice. |
| **Intermediate similarity** | Moderately similar. |
| **Low similarity** | Not similar. |
| **Insufficient data** | There was not enough speech to measure. Not a result. |

**Insufficient data** is not a weak answer — it is no answer. The minimums are
3 seconds of questioned speech and 10 seconds of reference speech, and below
those the number would be noise.

A score out of 1.00 is shown alongside the band. It is a similarity
measurement. It is **not** a percentage, not a probability, and not a
likelihood that the two recordings contain the same person — see
[section 12](#12-what-the-results-do-and-do-not-mean).

**Thresholds…** shows the exact values in force. They are conservative
starting points, not figures calibrated against your recordings, and they are
recorded in every exported report so a result can always be read against the
settings that produced it.

**Export report…** writes the result with everything needed to check it later:
the recordings, the models, the thresholds, and the caveats.

---

## 9. Comparison History

Every comparison this copy has run, with the recordings and settings used.
Useful for two things: finding a result you ran last week, and showing how a
conclusion was arrived at. **Export as CSV…** takes the lot out.

The caveats recorded at the time are kept with each entry — so a result cannot
later be read as cleaner than it was.

---

## 10. Live and Translate

**Live** transcribes an incoming stream as it arrives. Use a small model
(`base.en` or `small.en`) so it keeps up. **Chunk length** trades latency
against smoothness: shorter is more live but choppier. The transcript can be
saved to a file as it goes.

**Translate** translates pasted text, and can pull text out of an image or PDF
first (**Choose an image or PDF**) when the build includes OCR. The result can
be saved as a Word document. Like everything else here, it is offline.

---

## 11. When something goes wrong

**Start with System status.** It names what this copy can and cannot do, and
rules out half the possibilities in one look.

| What you see | What to do |
| --- | --- |
| Speaker separation fails, mentions the internet | The model cache in this build is not readable. Nothing on this machine will fix it; the build has to be redone. |
| Speech is missing from the transcript | Turn off **Skip silence** and run it again. |
| Two people came out as five | **Redo speaker separation…** with the right count. Next time, set the count before running. |
| A model is not in this build | **System status** lists the ones that are. Pick one of those. |
| It looks like it has frozen | Look for a dialog waiting for an answer — including on your other monitor. |
| Odd failures, files that will not open | Run `whispr.exe --verify`. A bundle damaged in transit fails differently every run. |

The **Status** tab holds the technical detail for anything that went wrong. It
is the right thing to copy when reporting a problem.

A long recording legitimately takes a long time. A 90-minute call on a bigger
model is an hour or more of work, and speaker separation adds to that. It is
not stuck.

---

## 12. What the results do and do not mean

This section is the most important one in this guide.

> Speaker similarity results produced by Whispers are investigative indicators
> intended to support lead development and analyst review. They are not
> forensic speaker identification, are not a biometric probability of identity,
> and should not be treated as proof that two recordings contain the same
> person.

In practice:

- **A High similarity band is a lead, not an identification.** It says this
  voice resembles that reference closely enough to be worth someone's
  attention. It does not say they are the same person, and it never will.
- **Never convert the score into a percentage or a probability.** "0.71" is a
  similarity measurement on a scale this tool defines. "71% likely to be him"
  is a different claim, one the tool has not made and cannot support.
- **Do not write "match", "confirmed", or "positive identification".** Write
  what the tool reported — the band — and what you intend to do about it.
- **The thresholds are starting points.** They were set conservatively, not
  calibrated against your recordings. Two results from builds with different
  thresholds are not directly comparable, which is why every report records
  the values it used.
- **A reference profile is only as good as what went into it.** A profile
  built from one short, poor sample will produce confident-looking numbers
  that mean very little. Quality and quantity of reference speech is the main
  thing under your control.
- **The transcript is a draft until you have checked it.** The tool is good,
  and on bad audio it still guesses. Anything you are going to rely on — a
  quote, a name, a number, a time — should be listened to before it leaves
  your hands.

Used this way the tool is a fast way to find what is worth a human's
attention in far more audio than a human could listen to. Used as an oracle,
it will eventually put a name on the wrong person's words.
