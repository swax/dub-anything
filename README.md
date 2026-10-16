# Dub Anything

> **Jellyfin experiment — deferred (2026-10-15).** This branch preserves the
> Jellyfin plugin, worker, updater, and their commit history. Dubbing takes hours
> and still needs active supervision for errors, exceptions, and directing
> decisions. An AI harness is the better place to manage that work for now, so
> `main` returns to direct, agent-led dubbing with the useful core fixes retained.
> This branch is a reference for possible future work, not the recommended setup.
> Its release feeds and automatic updates are no longer maintained.

Dub films and shows into another language **in each actor's own voice**, with
the original music and sound effects intact. It runs on your own PC, and an AI
coding agent directs the dub line by line, the way a dubbing director would.

You give it a video file. You get the same file back as
`my-show (dubbed).mkv`: the dub plays by default, the original audio is one
track away, and subtitles in both languages are included.

It dubs Japanese, Spanish, Korean and Chinese into English, and English into
Japanese, Spanish and Cantonese (yue). Need another language? Just ask: your agent adds the new
pair as part of the job. Once it works, please
[send it back](#other-languages-and-hardware) so everyone gets it.

## Demos

| A Japanese anime, dubbed into English | A Japanese film, dubbed into English |
|---|---|
| [![A Japanese anime, dubbed into English](https://img.youtube.com/vi/qXosYS-cTXk/maxresdefault.jpg)](https://www.youtube.com/watch?v=qXosYS-cTXk) | [![A Japanese film, dubbed into English](https://img.youtube.com/vi/2DsFAoTbQ8E/maxresdefault.jpg)](https://www.youtube.com/watch?v=2DsFAoTbQ8E) |

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
  side by side. What was measured is in [docs/research.md](docs/research.md).

## What you need

- **macOS (Apple Silicon M1/M2/M3/M4 or Intel)** with Metal (MPS) acceleration,
  or **a Windows PC with an NVIDIA GPU** with at least 6 GB of memory. A 16 GB
  card runs everything at once. (Linux should be a short port; see
  [below](#other-languages-and-hardware).)
- **Disk space:** about 40 GB for VoiceStudio, the models and the tools'
  environments, plus room to work: a movie in both voices used about
  85 GB until it was cleaned up.
- **An AI coding agent** that reads [AGENTS.md](AGENTS.md): Claude Code,
  Codex, Cursor or similar.
- **[VoiceStudio](https://github.com/debpalash/VoiceStudio)**, the open-source
  voice engine Dub Anything drives. The setup guide installs it.

## Getting started

1. **Set up once** with [docs/setup.md](docs/setup.md): VoiceStudio, its
   models, ffmpeg and uv.
2. **Open this folder in your agent** and ask, for example:

   > Dub D:\movies\my-show.mkv into English.

3. **Pick a voice** when it asks ([below](#choosing-a-voice)). Into Japanese
   or Spanish there's nothing to pick.
4. **Let it work.** The machines separate, transcribe, render and check while
   the agent directs each part as soon as it's drafted. It tells you if
   something needs you.
5. **Listen.** At the end you get the dubbed file and a short list of moments
   to check, with their times. Tell the agent what to change, and it
   re-renders just those parts.

The agent can't hear, so you're the ears: you judge the acting, the voices and
the mix, and you decide what may be cloned or shared.

The steps above are the normal direct workflow. Jellyfin is an optional way to
queue library videos with a headless director; its separate setup is in
[docs/jellyfin.md](docs/jellyfin.md). Direct dubs do not need the plugin, worker,
or updater. The [script guide](scripts/README.md) identifies each entry point.

## Choosing a voice

| | **Actor's voice** (default) | **American** |
|---|---|---|
| Sounds like | The actor speaking English, with their own accent | Native American English, converted toward each actor's voice |
| Best for | Live action | Anime and animation |
| Trade-offs | Accented English | Less like the actors in live action, a little less of the acting, and a few times slower to render |

Not sure? Ask for both. Each gets its own audio track, and you switch between
them in your player (the actor's voice plays by default). The American voice
has so far been used on Japanese sources.

Into Japanese or Spanish, the actor's voice is the only one: each actor speaks
the new language with their own accent.

## How long it takes

A short clip takes minutes. A movie in both voices takes about seven
hours of machine time on a laptop RTX 4090, nearly all of it on the GPU; the
actor's voice alone takes about half that. The agent directs while the renders
run, and on a long video it can split the directing across helper agents.

## Privacy

Your video and audio stay on your machine: the models run locally, and the
internet is only used to download them. Your agent does read the transcripts
and the video frames it looks at, so those go wherever your agent's model runs.

## Responsible use

Only dub media you have the rights to, and only clone voices with the consent
the use requires. A dub made from someone else's film or voice is for private
use unless you have permission.

## How it works

1. **Split** the video into ~3-minute parts at pauses in speech.
2. **Separate** the voices from the music and effects (Mel-Band Roformer).
3. **Transcribe** each part with Whisper large-v3, with two other recognisers
   as a second opinion, and burned-in captions where there are any.
4. **Cast** each line by matching voices across the whole show, checked
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

## Documentation

- [docs/jellyfin.md](docs/jellyfin.md): Jellyfin dashboard plugin, local queue,
  and headless Codex or Claude Code directors
- [docs/setup.md](docs/setup.md): installing everything, step by step
- [AGENTS.md](AGENTS.md): the agent's instructions, and a dub from start to
  finish
- [docs/directing.md](docs/directing.md): the directing workflow, every tool,
  and the gotchas
- [docs/languages/](docs/languages/README.md): notes for each language, and
  how to add one
- [docs/research.md](docs/research.md): what was measured and tried
- [docs/roadmap.md](docs/roadmap.md): what's next

## Running it without an agent

The scripts are plain Python, so you can drive a dub yourself and write the
directed scripts by hand. With VoiceStudio's Python
(`$py = "C:\git\VoiceStudio\.venv\Scripts\python.exe"`),
`scripts/dub_anything.py` runs the machine work of a whole dub:

```powershell
& $py scripts\dub_anything.py new D:\movies\my-show.mkv --voices both   # a job named my_show
& $py scripts\dub_anything.py run my_show --watch   # prep, then render each part marked ready
# write work\my_show-NN\script.json for each part, then:
& $py scripts\dub_anything.py ready my_show 01
& $py scripts\dub_anything.py status my_show
```

<details>
<summary>Or one step at a time</summary>

With the backend running (`scripts/backend.py start`):

```powershell
& $py scripts\split.py videos\episode.mkv                      # long video -> parts at pauses
& $py scripts\dub.py prep videos\episode-parts\episode-01.mp4  # upload, separate, transcribe
& $py scripts\draft.py episode-01                              # careful transcript, suggested cast
# write work\episode-01\script.json and the cast in work\voices.json
& $py scripts\dub.py generate episode-01 --match-level         # clone, time, mix, export
& $py scripts\lineqa.py episode-01                             # check each rendered line
& $py scripts\assemble.py episode --parts 1-4                  # one video with both audio tracks
& $py scripts\deliver.py D:\movies\episode.mkv actor=videos\episode-parts\dubbed\episode.dubbed.mp4
```

Some steps need their own environments (`separate.py`, `subs.py`,
`second.py`; see [AGENTS.md](AGENTS.md)).

</details>

The workflow, the script format and every tool are in
[docs/directing.md](docs/directing.md).

## Other languages and hardware

Ask for a language pair that isn't here yet, and your agent adds it as part of
the job. The voices aren't the limit: VoiceStudio clones voices in hundreds of
languages. What's new for each language is what the scripts know about it: the
phrases Whisper hallucinates in it, the recognisers that give second opinions,
and how QA compares a take with its line. That lives in one module,
`scripts/languages/<code>.py`. The agent writes it, then checks the pair on a
short test dub before adding it, as
[docs/languages/](docs/languages/README.md#adding-a-language) describes.

**Please send new languages back.** If your agent added a language or a pair,
open a pull request with its module, its notes (`docs/languages/<code>.md`)
and the pair's entry in `PAIRS`, so the next person gets it ready-made.

The models run on Metal MPS (on Apple Silicon macOS) or CUDA (on Windows/Linux)
and fall back to the CPU without it, slower. Video is encoded with
Apple VideoToolbox on macOS, NVENC on Windows/Linux when NVIDIA is present,
else x264. Platform differences (paths, desktop app runtime detection, GPU locks,
audio separation environments) are handled automatically across macOS and Windows.

## Contributing

Bug reports, ideas and pull requests are welcome, new languages most of all.
[docs/directing.md](docs/directing.md) explains how the pieces fit. Python is
formatted and linted with ruff (`uvx ruff format`, `uvx ruff check`), and
[docs/setup.md](docs/setup.md) shows how to turn on the pre-commit check.

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
