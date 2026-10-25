"""Styled subtitles (ASS): the dialogue at the bottom, and the translations of
on-screen text (a part's "signs") placed by the text they translate.

assemble.py writes, when any part has signs or subtitle_layout, <out>.en.ass
(dialogue and signs, the English track; .ja.ass for a dub into Japanese).
With signs, <out>.signs.ass holds them alone, for watching the dub without
dialogue subtitles. deliver.py then delivers Matroska: MP4's subtitles can't
be placed. Each language uses a font that has its letters (FONT in
languages/<code>.py), and full-width
characters (kana, kanji) are measured as such.

A sign, in the part's "signs" (times in the part, like its lines):
  {"start", "end", "text", "box": [x0, y0, x1, y1]}
the box as fractions of the picture (signs.py's candidates carry it), and
optionally
  "box_end": the box at the end, when the camera pans: the text moves with it,
           between the times in "seen" ([t0, t1], from the candidate) if
           given, else over the whole sign
  "place": where the translation goes: "over" the original (default), "above",
           "below", "left" or "right" of it, or "page": from the box's top
           left corner, left-aligned and wrapped in the box, for a page of text
  "size": font size relative to the usual (1.0); "\\n" in the text breaks a line
  "width": the width the text wraps in, as a fraction of the picture (default:
           the picture's), for a long text over a page or a panel
  "boxed": true puts the text on a dark box, for text over busy writing
A line of dialogue that would cover a sign shown with it moves to the top,
if the top is clear. Explicit subtitle_layout entries {start, end, top} override
that choice: top is the top-centre anchor as a fraction of the picture's height.
"""

import re
import unicodedata

from languages import LANGUAGES

DIALOGUE_SIZE = 0.054  # of the picture's height
SIGN_SIZE = 0.042
PLACES = {"over": 5, "above": 2, "below": 8, "left": 6, "right": 4, "page": 7}


def ems(line, latin):
    """A line's width in ems: a full-width character (kana, kanji) is one em, others `latin`."""
    return sum(1.0 if unicodedata.east_asian_width(c) in "WF" else latin for c in line)


def ass_time(t):
    cs = round(max(t, 0) * 100)
    return f"{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}"


def escape(text):
    """Plain text for an event: braces would start override tags."""
    text = text.replace("{", "(").replace("}", ")").replace("\\n", "\n")
    return re.sub(r"\s*\n\s*", r"\\N", text.strip())


def header(w, h, lang="en"):
    m, d, s = round(h * 0.03), round(h * DIALOGUE_SIZE), round(h * SIGN_SIZE)
    out = max(2, round(h / 360))
    font = LANGUAGES[lang].FONT
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Dialogue,{font},{d},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,{out},1,2,{m * 2},{m * 2},{round(h * 0.045)},1
Style: Sign,{font},{s},&H00FFFFFF,&H000000FF,&H00301A0A,&H80000000,-1,0,0,0,100,100,0,0,1,{out},1,5,{m},{m},{m},1
Style: SignBoxed,{font},{s},&H00FFFFFF,&H000000FF,&H30141414,&H00000000,0,0,0,0,100,100,0,0,3,{out * 3},0,5,{m},{m},{m},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def extent(lines, size, w, h, room=None):
    """An estimate of the text's (width, height): Arial bold averages about half
    an em a character (a Japanese character takes a whole em), and the renderer
    wraps a line wider than its room (the picture's width unless given)."""
    widths = [ems(l, 0.55) * size for l in lines]
    room = room or w - 2 * h * 0.03
    return min(max(widths), room), sum(-(-wd // room) or 1 for wd in widths) * size * 1.2


def anchor(box, place, size, lines, w, h, room=None):
    """(alignment, x, y) for text placed by a box, kept on screen."""
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    gap = size * 0.3
    an = PLACES[place]
    x, y = {
        5: ((x0 + x1) / 2, (y0 + y1) / 2),
        2: ((x0 + x1) / 2, y0 - gap),
        8: ((x0 + x1) / 2, y1 + gap),
        6: (x0 - gap, (y0 + y1) / 2),
        4: (x1 + gap, (y0 + y1) / 2),
        7: (x0, y0),
    }[an]
    m = h * 0.03
    tw, th = extent(lines, size, w, h, room)
    left = {4: x, 7: x, 6: x - tw}.get(an, x - tw / 2)
    top = {2: y - th, 7: y, 8: y}.get(an, y - th / 2)
    x += min(max(left, m), max(w - m - tw, m)) - left
    y += min(max(top, m), max(h - m - th, m)) - top
    return an, x, y


def room(sign, w):
    """The width a sign's text wraps in, if not the picture's."""
    if sign.get("width"):
        return sign["width"] * w
    return (sign["box"][2] - sign["box"][0]) * w if sign.get("place") == "page" else None


def layout(sign, w, h):
    """(size, text, lines, alignment, x, y) of a sign."""
    size = h * SIGN_SIZE * sign.get("size", 1.0)
    text = escape(sign["text"])
    lines = text.split("\\N")
    place = sign.get("place", "over")
    if place not in PLACES:
        raise ValueError(f"sign at {sign['start']:.2f}: place {place!r} is not one of {', '.join(PLACES)}")
    return (size, text, lines) + anchor(sign["box"], place, size, lines, w, h, room(sign, w))


def margins(an, x, width, w):
    """MarginL and MarginR that make a positioned text wrap within `width` around x."""
    left = {4: x, 7: x, 6: x - width}.get(an, x - width / 2)
    return max(0, round(left)), max(0, round(w - left - width))


def sign_rects(sign, w, h):
    """Where a sign's English is, and its original: [(x0, y0, x1, y1)] in
    pixels, at its start and its end."""
    size, _, lines, an, x, y = layout(sign, w, h)
    tw, th = extent(lines, size, w, h, room(sign, w))
    out = []
    for box in [sign["box"]] + ([sign["box_end"]] if sign.get("box_end") else []):
        _, x, y = anchor(box, sign.get("place", "over"), size, lines, w, h, room(sign, w))
        left = {4: x, 7: x, 6: x - tw}.get(an, x - tw / 2)
        top = {2: y - th, 7: y, 8: y}.get(an, y - th / 2)
        out.append((left, top, left + tw, top + th))
        out.append((box[0] * w, box[1] * h, box[2] * w, box[3] * h))
    return out


def dialogue_rect(text, w, h, top=False):
    """Where a line of dialogue shows, bottom centre (or top centre)."""
    size, m = h * DIALOGUE_SIZE, h * 0.045
    lines = escape(text).split("\\N")
    widths = [ems(l, 0.5) * size for l in lines]
    room = w - 4 * h * 0.03
    tw, th = min(max(widths), room), sum(-(-wd // room) or 1 for wd in widths) * size * 1.2
    y0 = m if top else h - m - th
    return (w - tw) / 2, y0, (w + tw) / 2, y0 + th


def meets(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def sign_event(sign, w, h):
    size, text, lines, an, x, y = layout(sign, w, h)
    tags = f"\\an{an}"
    end_box = sign.get("box_end")
    if end_box and max(abs(a - b) for a, b in zip(end_box, sign["box"])) > 0.01:
        _, x2, y2 = anchor(end_box, sign.get("place", "over"), size, lines, w, h, room(sign, w))
        when = ""
        if sign.get("seen"):
            t1, t2 = (max(0, round((t - sign["start"]) * 1000)) for t in sign["seen"])
            when = f",{t1},{t2}"
        tags += f"\\move({x:.0f},{y:.0f},{x2:.0f},{y2:.0f}{when})"
    else:
        tags += f"\\pos({x:.0f},{y:.0f})"
    if sign.get("size", 1.0) != 1.0:
        tags += f"\\fs{size:.0f}"
    ml, mr = margins(an, x, room(sign, w), w) if room(sign, w) else (0, 0)
    style = "SignBoxed" if sign.get("boxed") else "Sign"
    return f"Dialogue: 1,{ass_time(sign['start'])},{ass_time(sign['end'])},{style},,{ml},{mr},0,,{{{tags}}}{text}"


def write(path, w, h, dialogue=(), signs=(), lang="en", subtitle_layout=()):
    """dialogue: [(start, end, text)]; signs: sign dicts in the file's time; lang:
    the language they're in. A line of dialogue that would cover a sign shown with
    it moves to the top, if the top is clear. Timed subtitle_layout entries
    override that placement for intersecting cues; the last matching entry wins.
    Apply these to the cues after their reading time has been extended."""
    rects = [(s["start"], s["end"], sign_rects(s, w, h)) for s in signs]
    events = []
    for a, b, t in dialogue:
        near = [r for y, z, rs in rects if min(b, z) - max(a, y) > 0.05 for r in rs]
        up = any(meets(dialogue_rect(t, w, h), r) for r in near) and not any(
            meets(dialogue_rect(t, w, h, top=True), r) for r in near
        )
        tags = "{\\an8}" if up else ""
        explicit = next((s for s in reversed(subtitle_layout) if min(b, s["end"]) > max(a, s["start"])), None)
        if explicit is not None:
            tags = f"{{\\an8\\pos({w / 2:.0f},{h * explicit['top']:.0f})}}"
        events.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Dialogue,,0,0,0,," + tags + escape(t))
    events += [sign_event(s, w, h) for s in sorted(signs, key=lambda s: s["start"])]
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write(header(w, h, lang) + "\n".join(events) + "\n")
