# Dub Anything

Dub any video into another language **in each actor's own voice**, with
the original music and sound effects intact. It runs on your own PC, and an AI
coding agent directs the dub line by line, the way a dubbing director would.

You give it a video file and target language. You get the same file back as
`my-video (dubbed).mkv`: the dub plays by default, the original audio is one
track away, and subtitles in both languages are included.

Or ask for subtitles alone: the same agent-directed transcript and translation,
timed for reading, with text on screen translated where it appears, as
`my-video (subtitled).mkv` plus an `.srt` beside it. Subtitles need no voice
engine running.

## Demos

| Japanese animation, dubbed into English | Japanese live action, dubbed into English |
|---|---|
| [![Japanese animation, dubbed into English](https://img.youtube.com/vi/qXosYS-cTXk/maxresdefault.jpg)](https://www.youtube.com/watch?v=qXosYS-cTXk) | [![Japanese live action, dubbed into English](https://img.youtube.com/vi/2DsFAoTbQ8E/maxresdefault.jpg)](https://www.youtube.com/watch?v=2DsFAoTbQ8E) |

## What you need

A computer, one of:

- **macOS** (Apple Silicon or Intel)
- **Windows with an NVIDIA GPU** (6 GB or more)
- **Windows with an AMD Radeon GPU** (RDNA 3 or newer, Ryzen AI laptop graphics
  included)

Linux should be a short port; see [below](#other-languages-and-hardware).

An AI coding agent to direct the dub, and a plan to run it, one of:

- **Claude** Pro or Max, with Claude Code or the Claude desktop app
- **ChatGPT** Plus or Pro, with Codex
- **OpenCode** with an API key, or a local model that can read images

**Disk space:** about 40 GB, plus room to work (a long video in both voices
used about 85 GB until it was cleaned up).

Your agent installs [VoiceStudio](https://github.com/debpalash/VoiceStudio),
the voice engine, and the other prerequisites as part of the first run.

## Getting started

1. **Install/Open a desktop AI app**, such as ChatGPT, Claude or OpenCode,
   with access to your local files and a terminal. Turn on remote control so 
   you can monitor the dubbing process when you're away from your computer.
2. **Ask it to dub your video.** Give it this repository, your video's
   path and the language you want. For example:

   > Use https://github.com/swax/dub-anything to dub D:\videos\my-video.mkv
   > into English. Download the repository and install the prerequisites for me.

3. **Let the agent handle the rest.** It reads the project instructions,
   installs the tools, VoiceStudio and models, then directs, renders and checks
   the dub. It may ask you to sign in or accept a model license;
   after that, it's mostly hands off. The first run includes large downloads,
   and a long video can take hours.
4. **Watch the result.** You get the dubbed file and a short list of moments to
   check, with timestamps. If anything needs fixing, just ask the AI: tell it
   what went wrong and where, and it re-renders the affected parts.

Each actor keeps their native accent by default, using a faster engine with
more accurate voice cloning. You can specify a different accent in your request,
such as American English for an English dub.

## What makes it different

Most dubbing tools run a fixed pipeline: transcribe, translate, clone each
speaker from one clip, speed up whatever runs long. Dub Anything differs where
a listener notices:

- **An agent directs, line by line.** It works out who says each line from
  voice matching, video frames and the dialogue, fixes what speech recognition
  misheard, writes a translation timed to the actor, and triages every QA flag.
  It tells you which lines are guesses instead of hiding them.
- **Each line is acted like the original.** The cloned voice is prompted with
  the actor's own delivery of that very line, so a shout stays a shout and a
  murmur stays a murmur.
- **Each line takes the actor's time.** Every take is re-timed to end where the
  actor stops, so turns land where they did in the original, without dead air.
  Each line's length is predicted before rendering, so the translation is
  written to fill the time.
- **Every line is checked.** Each rendered line is transcribed back. A garbled
  one gets fresh takes, up to three rounds, and the best take is kept.
- **It reads the picture.** Burned-in captions become the transcript (and often
  name the speaker), and text on screen, like title cards and signs, can be
  translated as subtitles placed beside it.
- **The rest stays original.** Between lines you hear the original soundtrack,
  so songs, laughs and names stay the actor's own.
- **Chosen by ear.** The defaults were picked by listening to the alternatives
  side by side. What was measured is in [docs/research.md](docs/research.md). Feel free to continue fine tuning, and send back your improvements.

## How long it takes

A short clip takes minutes. A 90-minute video with the actors' native accents
took 4.1 hours of machine time on a laptop RTX 4090, nearly all of it on the
GPU: about half preparing the parts and two fifths rendering and checking the
voices. Adding an optional American English audio track brings the total to
about seven hours. The agent directs while the renders run, and on a long
video it can split the directing across helper agents.

A 16 GB NVIDIA card runs everything at once. Apple Silicon uses Metal (MPS)
acceleration. AMD runs through AMD's ROCm build of PyTorch and is slower: on a
laptop's integrated Radeon 890M, 15 minutes of video took about three hours of
machine time, so a 90-minute video would run overnight or longer. Integrated
graphics share the system's memory, so give them 32 GB of RAM or more.

Subtitles alone take about half the machine time of a dub, since no voice is
rendered or checked and the voice engine's own transcription is skipped. A
30-minute video took 41 minutes, two thirds of it reading the text in the
picture to translate it; without that step, it's about 15 minutes per half hour
of video. It was delivered 43 minutes after the job started, directing
included.

The agent's side takes a lot of tokens. Directing a 15-minute video (five
parts, about 190 lines) took about 175 agent turns: about 0.2M output tokens
and 0.4M new input tokens, plus about 110M input tokens read back from the
prompt cache, since every turn re-reads the conversation so far. A 90-minute
video has about six times the lines, which the agent shares out among helper
agents. On a smaller plan, expect to hit its usage limit partway through a long
video; the job keeps its progress, so ask the agent to carry on once the limit
resets.

## Privacy

Audio and video processing runs locally. Your AI provider may receive
transcripts and selected video frames as the agent directs and checks the dub. 
To run completely offline you need to use a local LLM for the director.

## Responsible use

Only dub media you have the rights to, and only clone voices with the consent
the use requires. A dub made from someone else's video or voice is for private
use unless you have permission.

<a id="how-it-works"></a>
<details>
<summary><strong>How it works</strong></summary>

1. **Split** the video into ~3-minute parts at pauses in speech.
2. **Separate** the voices from the music and effects (Mel-Band Roformer).
3. **Transcribe** each part with Whisper large-v3, with two other recognisers
   as a second opinion, and burned-in captions where there are any.
4. **Cast** each line by matching voices across the whole video, checked
   against video frames.
5. **Direct** (the agent): correct the transcript, translate, and time every
   line to the actor.
6. **Voice** each line. The actor's voice: OmniVoice clones the actor, prompted
   with their own delivery of that line. American: IndexTTS speaks native
   English following the actor's delivery, and Chatterbox converts it to the
   actor's voice.
7. **Fit, mix and check.** Each take is re-timed to end where the actor stops,
   mixed over the original background, and transcribed back to catch garbled
   words.
8. **Deliver** into your video file. Nothing is re-encoded, and every stream is
   checked against the original before the file is replaced.

<details>
<summary>The whole pipeline in one diagram</summary>

![How a dub gets made](docs/process.png)

The diagram is [docs/process.jg](docs/process.jg), a
[Jumpgate](https://github.com/swax/jumpgate) file. It is rendered with
`npm run render -- <path>/docs/process.jg --width 1800 --height 2700` from
Jumpgate's `packages/jumpgate`.

</details>

</details>

<a id="documentation"></a>
<details>
<summary><strong>Documentation</strong></summary>

- [scripts/README.md](scripts/README.md): script entry points and shared tools
- [docs/setup.md](docs/setup.md): setup instructions for agents
- [AGENTS.md](AGENTS.md): the agent's instructions, and a dub from start to
  finish
- [docs/directing.md](docs/directing.md): the directing workflow, every tool,
  and the gotchas
- [docs/languages/](docs/languages/README.md): notes for each language, and
  how to add one
- [docs/research.md](docs/research.md): what was measured and tried
- [docs/roadmap.md](docs/roadmap.md): what's next

</details>

<a id="running-it-without-an-agent"></a>
<details>
<summary><strong>Running it without an agent</strong></summary>

The scripts are plain Python, so you can drive a dub yourself and write the
directed scripts by hand. With VoiceStudio's Python
(`$py = "C:\git\VoiceStudio\.venv\Scripts\python.exe"`),
`scripts/dub_anything.py` runs the machine work of a whole dub:

```powershell
& $py scripts\dub_anything.py new D:\videos\my-video.mkv --voices both   # a job named my_video
& $py scripts\dub_anything.py run my_video --watch   # prep, then render each part marked ready
# write work\my_video-NN\script.json for each part, then:
& $py scripts\dub_anything.py ready my_video 01
& $py scripts\dub_anything.py status my_video
```

<details>
<summary>Or one step at a time</summary>

With the backend running (`scripts/backend.py start`):

```powershell
& $py scripts\split.py videos\video.mkv                    # long video -> parts at pauses
& $py scripts\dub.py prep videos\video-parts\video-01.mp4  # upload, separate, transcribe
& $py scripts\draft.py video-01                            # careful transcript, suggested cast
# write work\video-01\script.json and the cast in work\voices.json
& $py scripts\dub.py generate video-01 --match-level       # clone, time, mix, export
& $py scripts\lineqa.py video-01                           # check each rendered line
& $py scripts\assemble.py video --parts 1-4                # one video with both audio tracks
& $py scripts\deliver.py D:\videos\video.mkv actor=videos\video-parts\dubbed\video.dubbed.mp4
```

Some steps need their own environments (`separate.py`, `subs.py`,
`second.py`; see [AGENTS.md](AGENTS.md)).

</details>

The workflow, the script format and every tool are in
[docs/directing.md](docs/directing.md).

</details>

<a id="other-languages-and-hardware"></a>
<details>
<summary><strong>Other languages and hardware</strong></summary>

Ask for a language pair that isn't here yet, and your agent adds it as part of
the job. The voices aren't the limit: VoiceStudio clones voices in hundreds of
languages. What's new for each language is what the scripts know about it: the
phrases Whisper hallucinates in it, the recognisers that give second opinions,
and how QA compares a take with its line. That lives in one module,
`scripts/languages/<code>.py`. The agent writes it, then checks the pair on a
short test dub before adding it, as
[docs/languages/](docs/languages/README.md#adding-a-language) describes.

**Share new languages if you'd like.** Ask your agent to open a pull request
with its module, its notes (`docs/languages/<code>.md`) and the pair's entry in
`PAIRS`, so the next person gets it ready-made.

The models run on Metal MPS (on Apple Silicon macOS), CUDA (on Windows/Linux)
or ROCm (AMD Radeon on Windows; Whisper stays on the CPU there) and fall back
to the CPU without them, slower. Video is encoded with
Apple VideoToolbox on macOS, NVENC on Windows/Linux when NVIDIA is present,
else x264. Platform differences (paths, desktop app runtime detection, GPU locks,
audio separation environments) are handled automatically across macOS and Windows.

</details>

## Contributing

Bug reports, ideas and pull requests are welcome, new languages most of all.
If your AI fixes a bug, improves the workflow or adds a language, you can ask it
to send the changes back as a pull request. Sharing is optional.
[docs/directing.md](docs/directing.md) explains how the pieces fit. Python is
formatted and linted with ruff (`uvx ruff format`, `uvx ruff check`), and
the [agent setup guide](docs/setup.md) shows how to turn on the pre-commit check.

`videos/` and `work/` are git-ignored so that source media, transcripts and
voice references stay out of the repository. Don't force-add them, and use
invented examples in docs and commit messages.

## Built on

Dub Anything is direction and glue around excellent open-source work:

- [VoiceStudio](https://github.com/debpalash/VoiceStudio): the voice backend
  (dubbing jobs, cloning, mixing)
- [OmniVoice](https://github.com/k2-fsa/OmniVoice): the actor's-voice clones
- [IndexTTS](https://github.com/index-tts/index-tts): native English for the
  American voice
- [Chatterbox](https://github.com/resemble-ai/chatterbox): voice conversion to
  each actor's timbre
- [Whisper](https://github.com/openai/whisper), through
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper): transcription
- [pyannote](https://github.com/pyannote/pyannote-audio) and
  [WeSpeaker](https://github.com/wenet-e2e/wespeaker): who speaks when, and
  voice matching
- [Silero VAD](https://github.com/snakers4/silero-vad): where speech really
  starts and stops
- Mel-Band Roformer, through
  [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator):
  voices apart from the music
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): the second-opinion
  recognisers
- [RapidOCR](https://github.com/RapidAI/RapidOCR) with
  [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)'s PP-OCRv5: reading
  captions and signs
- [FFmpeg](https://ffmpeg.org): cutting, mixing and muxing

## License

Dub Anything is MIT-licensed ([LICENSE](LICENSE)). It talks to VoiceStudio over
HTTP and doesn't include its code. VoiceStudio is AGPL-3.0, and so is
`patches/voicestudio-local-fixes.patch`, a change to VoiceStudio's source. The
models have their own licenses: pyannote is gated on Hugging Face, the
Sortformer weights are CC-BY-NC, and Chatterbox is MIT. Check each one before
any commercial use.
