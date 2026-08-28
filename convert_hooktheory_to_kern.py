#!/usr/bin/env python3
"""Convert Sheet Sage Hooktheory JSON annotations into Humdrum **kern files.

The converter writes one two-spine Humdrum file per song:

* ``**kern`` for the monophonic melody, including clef, key, meter, barlines,
  rests, and ties where an input event must be split.
* ``**mxhm`` for chord labels derived from Hooktheory root-position intervals.

The input file can be the compressed ``Hooktheory.json.gz`` from
https://github.com/chrisdonahue/sheetsage-data/tree/main/hooktheory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


TICKS_PER_BEAT = 960
DEFAULT_OUTPUT_DIR = "krn"

LETTERS = ("C", "D", "E", "F", "G", "A", "B")
NATURAL_PITCH_CLASSES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

MAJOR = (2, 2, 1, 2, 2, 2)
NATURAL_MINOR = (2, 1, 2, 2, 1, 2)
MIXOLYDIAN = (2, 2, 1, 2, 2, 1)
DORIAN = (2, 1, 2, 2, 2, 1)
LYDIAN = (2, 2, 2, 1, 2, 2)
PHRYGIAN = (1, 2, 2, 2, 1, 2)
LOCRIAN = (1, 2, 2, 1, 2, 2)
HARMONIC_MINOR = (2, 1, 2, 2, 1, 3)
PHRYGIAN_DOMINANT = (1, 3, 1, 2, 1, 2)

MODE_NAMES = {
    MAJOR: "major",
    NATURAL_MINOR: "minor",
    MIXOLYDIAN: "mixolydian",
    DORIAN: "dorian",
    LYDIAN: "lydian",
    PHRYGIAN: "phrygian",
    LOCRIAN: "locrian",
    HARMONIC_MINOR: "harmonic minor",
    PHRYGIAN_DOMINANT: "phrygian dominant",
}


@dataclass(frozen=True)
class Meter:
    beat: Fraction
    beats_per_bar: int
    beat_unit: int


@dataclass(frozen=True)
class KeyContext:
    beat: Fraction
    tonic_pitch_class: int
    pattern: tuple[int, ...]
    tonic_letter: str
    tonic_accidental: int
    scale_accidentals: dict[str, int]
    signature_accidentals: dict[str, int]
    scale_spellings_by_pc: dict[int, tuple[str, int]]

    @property
    def mode(self) -> str:
        return MODE_NAMES.get(self.pattern, "mode")

    @property
    def prefers_flats(self) -> bool:
        flats = sum(1 for value in self.signature_accidentals.values() if value < 0)
        sharps = sum(1 for value in self.signature_accidentals.values() if value > 0)
        return flats > sharps

    @property
    def prefers_sharps(self) -> bool:
        flats = sum(1 for value in self.signature_accidentals.values() if value < 0)
        sharps = sum(1 for value in self.signature_accidentals.values() if value > 0)
        return sharps >= flats


@dataclass(frozen=True)
class MelodyEvent:
    onset: Fraction
    offset: Fraction
    octave: int
    pitch_class: int


@dataclass(frozen=True)
class HarmonyEvent:
    onset: Fraction
    offset: Fraction
    root_pitch_class: int
    root_position_intervals: tuple[int, ...]
    inversion: int


def quantize_beat(value: Any) -> Fraction:
    """Convert a JSON beat value to a stable rational grid."""

    return Fraction(round(float(value) * TICKS_PER_BEAT), TICKS_PER_BEAT)


def read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize_filename_part(value: Any, fallback: str, *, lowercase: bool = True) -> str:
    text = str(value or "").strip()
    if lowercase:
        text = text.lower()
    pattern = r"[^a-z0-9._-]+" if lowercase else r"[^A-Za-z0-9._-]+"
    text = re.sub(pattern, "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-.")
    return text or fallback


def humdrum_text(value: Any) -> str:
    text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def accidental_string(accidental: int) -> str:
    if accidental > 0:
        return "#" * accidental
    if accidental < 0:
        return "-" * abs(accidental)
    return ""


def accidental_for_letter(letter: str, pitch_class: int) -> int:
    natural = NATURAL_PITCH_CLASSES[letter]
    candidates = [pitch_class - natural + (12 * offset) for offset in range(-2, 3)]
    return min(candidates, key=lambda value: (abs(value), value))


def complete_scale(pattern: tuple[int, ...]) -> list[int]:
    pitch_classes = [0]
    current = 0
    for interval in pattern:
        current += interval
        pitch_classes.append(current % 12)
    return pitch_classes[:7]


def spell_scale(
    tonic_pitch_class: int, pattern: tuple[int, ...]
) -> tuple[str, int, dict[str, int], dict[int, tuple[str, int]]]:
    """Choose a readable diatonic spelling for a tonic and scale pattern."""

    scale = complete_scale(pattern)
    best: tuple[int, str, int, dict[str, int], dict[int, tuple[str, int]]] | None = None

    for tonic_letter_index, tonic_letter in enumerate(LETTERS):
        tonic_accidental = accidental_for_letter(tonic_letter, tonic_pitch_class)
        if (NATURAL_PITCH_CLASSES[tonic_letter] + tonic_accidental) % 12 != tonic_pitch_class:
            continue
        if abs(tonic_accidental) > 2:
            continue

        scale_accidentals: dict[str, int] = {}
        spellings_by_pc: dict[int, tuple[str, int]] = {}
        ok = True

        for degree, relative_pc in enumerate(scale):
            letter = LETTERS[(tonic_letter_index + degree) % len(LETTERS)]
            target_pc = (tonic_pitch_class + relative_pc) % 12
            accidental = accidental_for_letter(letter, target_pc)
            if abs(accidental) > 2:
                ok = False
                break
            scale_accidentals[letter] = accidental
            spellings_by_pc[target_pc] = (letter, accidental)

        if not ok:
            continue

        max_abs = max(abs(value) for value in scale_accidentals.values())
        total_abs = sum(abs(value) for value in scale_accidentals.values())
        mixed_direction = any(value > 0 for value in scale_accidentals.values()) and any(
            value < 0 for value in scale_accidentals.values()
        )
        tonic_penalty = abs(tonic_accidental) * 5
        score = (max_abs * 1000) + (total_abs * 10) + (100 if mixed_direction else 0) + tonic_penalty
        candidate = (score, tonic_letter, tonic_accidental, scale_accidentals, spellings_by_pc)

        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        letter = CHORD_SHARP_NAMES[tonic_pitch_class].replace("#", "")
        accidental = 1 if "#" in CHORD_SHARP_NAMES[tonic_pitch_class] else 0
        return letter, accidental, {letter: accidental}, {tonic_pitch_class: (letter, accidental)}

    _, tonic_letter, tonic_accidental, scale_accidentals, spellings_by_pc = best
    return tonic_letter, tonic_accidental, scale_accidentals, spellings_by_pc


def make_key_context(raw: dict[str, Any]) -> KeyContext:
    pattern = tuple(int(value) for value in raw.get("scale_degree_intervals") or MAJOR)
    tonic_pitch_class = int(raw.get("tonic_pitch_class", 0)) % 12

    tonic_letter, tonic_accidental, scale_accidentals, spellings_by_pc = spell_scale(
        tonic_pitch_class, pattern
    )

    signature_pattern = NATURAL_MINOR if pattern == HARMONIC_MINOR else pattern
    _, _, signature_accidentals, _ = spell_scale(tonic_pitch_class, signature_pattern)

    return KeyContext(
        beat=quantize_beat(raw.get("beat", 0)),
        tonic_pitch_class=tonic_pitch_class,
        pattern=pattern,
        tonic_letter=tonic_letter,
        tonic_accidental=tonic_accidental,
        scale_accidentals=scale_accidentals,
        signature_accidentals=signature_accidentals,
        scale_spellings_by_pc=spellings_by_pc,
    )


def key_signature_token(context: KeyContext) -> str:
    sharps_order = ("F", "C", "G", "D", "A", "E", "B")
    flats_order = ("B", "E", "A", "D", "G", "C", "F")
    parts: list[str] = []

    for letter in sharps_order:
        accidental = context.signature_accidentals.get(letter, 0)
        if accidental > 0:
            parts.append(letter.lower() + accidental_string(accidental))

    for letter in flats_order:
        accidental = context.signature_accidentals.get(letter, 0)
        if accidental < 0:
            parts.append(letter.lower() + accidental_string(accidental))

    return "*k[" + "".join(parts) + "]"


def key_designation_token(context: KeyContext) -> str:
    tonic = context.tonic_letter + accidental_string(context.tonic_accidental)
    third = complete_scale(context.pattern)[2] if len(complete_scale(context.pattern)) > 2 else 4
    if third == 3:
        tonic = tonic.lower()
    return f"*{tonic}:"


def select_context_at_beat(contexts: list[Any], beat: Fraction) -> Any:
    current = contexts[0]
    for context in contexts:
        if context.beat <= beat:
            current = context
        else:
            break
    return current


def pitch_spelling(pitch_class: int, context: KeyContext) -> tuple[str, int, bool]:
    pitch_class %= 12
    if pitch_class in context.scale_spellings_by_pc:
        letter, accidental = context.scale_spellings_by_pc[pitch_class]
        needs_natural = accidental == 0 and context.signature_accidentals.get(letter, 0) != 0
        return letter, accidental, needs_natural

    candidates: list[tuple[int, str, int, bool]] = []
    for letter in LETTERS:
        accidental = accidental_for_letter(letter, pitch_class)
        if abs(accidental) > 2:
            continue
        scale_accidental = context.scale_accidentals.get(letter, 0)
        key_accidental = context.signature_accidentals.get(letter, 0)
        alteration_distance = abs(accidental - scale_accidental)
        style_penalty = 0
        if accidental < 0 and context.prefers_sharps:
            style_penalty = 1
        elif accidental > 0 and context.prefers_flats:
            style_penalty = 1
        needs_natural = accidental == 0 and key_accidental != 0
        score = (alteration_distance * 100) + (abs(accidental) * 10) + style_penalty
        candidates.append((score, letter, accidental, needs_natural))

    if not candidates:
        return "C", 0, False

    _, letter, accidental, needs_natural = min(candidates, key=lambda item: item[0])
    return letter, accidental, needs_natural


def kern_pitch(pitch_class: int, octave: int, context: KeyContext) -> str:
    letter, accidental, needs_natural = pitch_spelling(pitch_class, context)
    if octave >= 0:
        base = letter.lower() * (octave + 1)
    else:
        base = letter.upper() * abs(octave)

    if accidental == 0 and needs_natural:
        return base + "n"
    return base + accidental_string(accidental)


def duration_candidates(beat_unit: int) -> list[tuple[int, str]]:
    ticks_per_whole = beat_unit * TICKS_PER_BEAT
    candidates: dict[int, tuple[str, tuple[int, int, int]]] = {}
    max_reciprocal = ticks_per_whole

    for reciprocal in range(1, max_reciprocal + 1):
        for dots in range(4):
            dotted_factor = sum(Fraction(1, 2**index) for index in range(dots + 1))
            ticks = Fraction(ticks_per_whole, reciprocal) * dotted_factor
            if ticks.denominator != 1 or ticks.numerator < 1:
                continue
            token = f"{reciprocal}{'.' * dots}"
            # Prefer fewer dots and smaller reciprocal values for equal tick spans.
            preference = (dots, reciprocal, len(token))
            existing = candidates.get(ticks.numerator)
            if existing is None or preference < existing[1]:
                candidates[ticks.numerator] = (token, preference)

    return sorted(
        ((ticks, token_pref[0]) for ticks, token_pref in candidates.items()),
        key=lambda item: item[0],
        reverse=True,
    )


_DURATION_CANDIDATE_CACHE: dict[int, list[tuple[int, str]]] = {}


def duration_token_from_ticks(ticks: int, beat_unit: int) -> str | None:
    ticks_per_whole = beat_unit * TICKS_PER_BEAT
    duration = Fraction(ticks, ticks_per_whole)
    for dots in range(4):
        dotted_factor = sum(Fraction(1, 2**index) for index in range(dots + 1))
        reciprocal = dotted_factor / duration
        if reciprocal.denominator == 1 and reciprocal.numerator > 0:
            return f"{reciprocal.numerator}{'.' * dots}"
    return None


def decompose_duration(beat_span: Fraction, beat_unit: int) -> list[tuple[Fraction, str]]:
    ticks_fraction = beat_span * TICKS_PER_BEAT
    if ticks_fraction.denominator != 1:
        raise ValueError(f"Duration is outside the {TICKS_PER_BEAT} PPQ grid: {beat_span}")

    remaining = ticks_fraction.numerator
    if remaining <= 0:
        return []

    direct = duration_token_from_ticks(remaining, beat_unit)
    if direct is not None:
        return [(beat_span, direct)]

    candidates = _DURATION_CANDIDATE_CACHE.setdefault(beat_unit, duration_candidates(beat_unit))
    pieces: list[tuple[Fraction, str]] = []

    while remaining > 0:
        direct = duration_token_from_ticks(remaining, beat_unit)
        if direct is not None:
            piece_ticks = remaining
            token = direct
        else:
            piece_ticks = 0
            token = ""
            for candidate_ticks, candidate_token in candidates:
                if candidate_ticks <= remaining:
                    piece_ticks = candidate_ticks
                    token = candidate_token
                    break
            if piece_ticks == 0:
                piece_ticks = 1
                token = str(beat_unit * TICKS_PER_BEAT)

        pieces.append((Fraction(piece_ticks, TICKS_PER_BEAT), token))
        remaining -= piece_ticks

    return pieces


CHORD_SHARP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHORD_FLAT_NAMES = ("C", "D-", "D", "E-", "E", "F", "G-", "G", "A-", "A", "B-", "B")


def chord_pitch_name(pitch_class: int, context: KeyContext) -> str:
    pitch_class %= 12
    if pitch_class in context.scale_spellings_by_pc:
        letter, accidental = context.scale_spellings_by_pc[pitch_class]
        return letter + accidental_string(accidental)

    names = CHORD_FLAT_NAMES if context.prefers_flats else CHORD_SHARP_NAMES
    return names[pitch_class]


CHORD_QUALITY_BY_INTERVALS = {
    (): "",
    (4,): "",
    (3,): "m",
    (5,): "sus4",
    (7,): "5",
    (4, 7): "",
    (3, 7): "m",
    (3, 6): "dim",
    (4, 8): "aug",
    (5, 7): "sus4",
    (2, 7): "sus2",
    (4, 7, 10): "7",
    (4, 7, 11): "maj7",
    (3, 7, 10): "m7",
    (3, 7, 11): "mMaj7",
    (3, 6, 9): "dim7",
    (3, 6, 10): "m7b5",
    (4, 8, 10): "aug7",
    (4, 8, 11): "augMaj7",
    (5, 7, 10): "7sus4",
    (5, 7, 11): "maj7sus4",
    (2, 7, 10): "7sus2",
    (2, 7, 11): "maj7sus2",
    (4, 7, 10, 14): "9",
    (4, 7, 11, 14): "maj9",
    (3, 7, 10, 14): "m9",
    (3, 6, 10, 14): "m9b5",
    (5, 7, 10, 14): "9sus4",
    (4, 7, 10, 14, 17): "11",
    (3, 7, 10, 14, 17): "m11",
    (4, 7, 11, 14, 18): "maj11#11",
    (4, 7, 10, 14, 17, 21): "13",
    (3, 7, 10, 14, 17, 21): "m13",
    (4, 7, 11, 14, 18, 21): "maj13#11",
}


def cumulative_intervals(interval_steps: tuple[int, ...]) -> tuple[int, ...]:
    total = 0
    result = []
    for step in interval_steps:
        total += step
        result.append(total)
    return tuple(result)


def chord_quality(interval_steps: tuple[int, ...]) -> str:
    intervals = cumulative_intervals(interval_steps)
    normalized = tuple(interval % 24 for interval in intervals)
    mapped = CHORD_QUALITY_BY_INTERVALS.get(normalized)
    if mapped is not None:
        return mapped

    pitch_classes = sorted({interval % 12 for interval in intervals})
    mapped = CHORD_QUALITY_BY_INTERVALS.get(tuple(pitch_classes))
    if mapped is not None:
        return mapped

    if not pitch_classes:
        return ""

    interval_text = ",".join(str(interval) for interval in pitch_classes)
    return f"add({interval_text})"


def chord_symbol(harmony: HarmonyEvent, context: KeyContext) -> str:
    root = chord_pitch_name(harmony.root_pitch_class, context)
    suffix = chord_quality(harmony.root_position_intervals)
    symbol = root + suffix

    chord_tones = [0]
    chord_tones.extend(cumulative_intervals(harmony.root_position_intervals))
    if harmony.inversion > 0 and harmony.inversion < len(chord_tones):
        bass_pc = (harmony.root_pitch_class + chord_tones[harmony.inversion]) % 12
        bass = chord_pitch_name(bass_pc, context)
        if bass != root:
            symbol += f"/{bass}"

    return symbol


def parse_melody(annotation: dict[str, Any], total_beats: Fraction) -> list[MelodyEvent]:
    events: list[MelodyEvent] = []
    for raw in annotation.get("melody") or []:
        onset = max(Fraction(0), min(total_beats, quantize_beat(raw["onset"])))
        offset = max(Fraction(0), min(total_beats, quantize_beat(raw["offset"])))
        if offset <= onset:
            continue
        events.append(
            MelodyEvent(
                onset=onset,
                offset=offset,
                octave=int(raw.get("octave", 0)),
                pitch_class=int(raw.get("pitch_class", 0)) % 12,
            )
        )
    return sorted(events, key=lambda event: (event.onset, event.offset))


def parse_harmony(annotation: dict[str, Any], total_beats: Fraction) -> list[HarmonyEvent]:
    events: list[HarmonyEvent] = []
    for raw in annotation.get("harmony") or []:
        onset = max(Fraction(0), min(total_beats, quantize_beat(raw["onset"])))
        offset = max(Fraction(0), min(total_beats, quantize_beat(raw["offset"])))
        if offset <= onset:
            continue
        events.append(
            HarmonyEvent(
                onset=onset,
                offset=offset,
                root_pitch_class=int(raw.get("root_pitch_class", 0)) % 12,
                root_position_intervals=tuple(
                    int(value) for value in raw.get("root_position_intervals") or []
                ),
                inversion=int(raw.get("inversion", 0)),
            )
        )
    return sorted(events, key=lambda event: (event.onset, event.offset))


def parse_meters(annotation: dict[str, Any]) -> list[Meter]:
    meters = []
    for raw in annotation.get("meters") or []:
        meters.append(
            Meter(
                beat=quantize_beat(raw.get("beat", 0)),
                beats_per_bar=int(raw.get("beats_per_bar", 4)),
                beat_unit=int(raw.get("beat_unit", 4)),
            )
        )
    if not meters:
        meters.append(Meter(beat=Fraction(0), beats_per_bar=4, beat_unit=4))
    return sorted(meters, key=lambda meter: meter.beat)


def parse_keys(annotation: dict[str, Any]) -> list[KeyContext]:
    contexts = [make_key_context(raw) for raw in annotation.get("keys") or []]
    if not contexts:
        contexts.append(make_key_context({"beat": 0, "tonic_pitch_class": 0, "scale_degree_intervals": MAJOR}))
    return sorted(contexts, key=lambda context: context.beat)


def barline_beats(meters: list[Meter], total_beats: Fraction) -> list[Fraction]:
    beats = {Fraction(0)}
    for index, meter in enumerate(meters):
        segment_start = max(Fraction(0), meter.beat)
        segment_end = total_beats
        if index + 1 < len(meters):
            segment_end = min(segment_end, meters[index + 1].beat)

        current = segment_start
        while current < segment_end:
            beats.add(current)
            current += meter.beats_per_bar

    return sorted(beat for beat in beats if Fraction(0) <= beat <= total_beats)


def metadata_records(song_id: str, record: dict[str, Any]) -> list[str]:
    hooktheory = record.get("hooktheory") or {}
    youtube = record.get("youtube") or {}
    urls = hooktheory.get("urls") or {}
    annotations = record.get("annotations") or {}

    rows = [
        "!!!COM: " + humdrum_text(hooktheory.get("artist", "")),
        "!!!OTL: " + humdrum_text(hooktheory.get("song", "")),
        "!!!HTI: " + humdrum_text(hooktheory.get("id", song_id)),
        "!!!HTU: " + humdrum_text(urls.get("song", "")),
        "!!!HTC: " + humdrum_text(urls.get("clip", "")),
        "!!!YTI: " + humdrum_text(youtube.get("id", "")),
        "!!!YTU: " + humdrum_text(youtube.get("url", "")),
        "!!!SPL: " + humdrum_text(record.get("split", "")),
        "!!!ONB: " + humdrum_text(annotations.get("num_beats", "")),
    ]

    if hooktheory.get("annotators"):
        rows.append("!!!HTA: " + humdrum_text("; ".join(hooktheory["annotators"])))
    if record.get("tags"):
        rows.append("!!!TAG: " + humdrum_text("; ".join(record["tags"])))
    if youtube.get("duration") is not None:
        rows.append("!!!YTD: " + humdrum_text(youtube["duration"]))

    return rows


def format_kern_data_token(
    duration: str,
    melody: MelodyEvent | None,
    segment_start: Fraction,
    segment_end: Fraction,
    context: KeyContext,
) -> str:
    if melody is None:
        return duration + "r"

    pitch = kern_pitch(melody.pitch_class, melody.octave, context)
    tie = ""
    if melody.onset < segment_start and segment_end < melody.offset:
        tie = "_"
    elif melody.onset == segment_start and segment_end < melody.offset:
        tie = "["
    elif melody.onset < segment_start and segment_end == melody.offset:
        tie = "]"

    return duration + pitch + tie


def convert_record(song_id: str, record: dict[str, Any]) -> str:
    annotation = record.get("annotations") or {}
    total_beats = quantize_beat(annotation.get("num_beats", 0))

    meters = parse_meters(annotation)
    keys = parse_keys(annotation)
    melody_events = parse_melody(annotation, total_beats)
    harmony_events = parse_harmony(annotation, total_beats)

    timepoints = {Fraction(0), total_beats}
    timepoints.update(event.onset for event in melody_events)
    timepoints.update(event.offset for event in melody_events)
    timepoints.update(event.onset for event in harmony_events)
    timepoints.update(event.offset for event in harmony_events)
    timepoints.update(meter.beat for meter in meters)
    timepoints.update(context.beat for context in keys)

    barlines = set(barline_beats(meters, total_beats))
    timepoints.update(barlines)

    sorted_timepoints = sorted(beat for beat in timepoints if Fraction(0) <= beat <= total_beats)
    if not sorted_timepoints or sorted_timepoints[0] != 0:
        sorted_timepoints.insert(0, Fraction(0))

    rows = metadata_records(song_id, record)
    rows.extend(["**kern\t**mxhm", "*clefG2\t*"])

    first_key = select_context_at_beat(keys, Fraction(0))
    first_meter = select_context_at_beat(meters, Fraction(0))
    rows.append(f"{key_signature_token(first_key)}\t*")
    rows.append(f"{key_designation_token(first_key)}\t*")
    rows.append(f"*M{first_meter.beats_per_bar}/{first_meter.beat_unit}\t*")

    harmony_by_onset: dict[Fraction, list[HarmonyEvent]] = {}
    for event in harmony_events:
        harmony_by_onset.setdefault(event.onset, []).append(event)

    meter_changes = {meter.beat: meter for meter in meters}
    key_changes = {context.beat: context for context in keys}

    measure_number = 1
    melody_index = 0

    for index, start in enumerate(sorted_timepoints[:-1]):
        end = sorted_timepoints[index + 1]
        if end <= start:
            continue

        if start in barlines:
            rows.append(f"={measure_number}\t={measure_number}")
            measure_number += 1

        if start != 0 and start in key_changes:
            context = key_changes[start]
            rows.append(f"{key_signature_token(context)}\t*")
            rows.append(f"{key_designation_token(context)}\t*")

        if start != 0 and start in meter_changes:
            meter = meter_changes[start]
            rows.append(f"*M{meter.beats_per_bar}/{meter.beat_unit}\t*")

        while melody_index < len(melody_events) and melody_events[melody_index].offset <= start:
            melody_index += 1

        melody: MelodyEvent | None = None
        if melody_index < len(melody_events):
            candidate = melody_events[melody_index]
            if candidate.onset <= start < candidate.offset:
                melody = candidate

        context = select_context_at_beat(keys, start)
        meter = select_context_at_beat(meters, start)
        pieces = decompose_duration(end - start, meter.beat_unit)
        piece_start = start

        for piece_index, (piece_span, duration) in enumerate(pieces):
            piece_end = piece_start + piece_span
            chord = "."
            if piece_index == 0 and start in harmony_by_onset:
                chord = chord_symbol(harmony_by_onset[start][0], context)
            kern = format_kern_data_token(duration, melody, piece_start, piece_end, context)
            rows.append(f"{kern}\t{chord}")
            piece_start = piece_end

    if total_beats != 0:
        if total_beats in key_changes:
            context = key_changes[total_beats]
            rows.append(f"{key_signature_token(context)}\t*")
            rows.append(f"{key_designation_token(context)}\t*")

        if total_beats in meter_changes:
            meter = meter_changes[total_beats]
            rows.append(f"*M{meter.beats_per_bar}/{meter.beat_unit}\t*")

    rows.append("==\t==")
    rows.append("*-\t*-")
    return "\n".join(rows) + "\n"


def output_filename(song_id: str, record: dict[str, Any]) -> str:
    hooktheory = record.get("hooktheory") or {}
    artist = sanitize_filename_part(hooktheory.get("artist"), "unknown-artist")
    song = sanitize_filename_part(hooktheory.get("song"), "unknown-song")
    clean_id = sanitize_filename_part(song_id, "unknown-id", lowercase=False)
    return f"{artist}_{song}_{clean_id}.krn"


def unique_output_path(output_dir: Path, filename: str, used_names: set[str]) -> Path:
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    candidate = filename
    counter = 2

    while candidate.casefold() in used_names:
        candidate = f"{stem}__{counter}{suffix}"
        counter += 1

    used_names.add(candidate.casefold())
    return output_dir / candidate


def convert_all(input_path: Path, output_dir: Path, limit: int | None = None) -> tuple[int, list[str]]:
    data = read_json(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    used_names: set[str] = set()
    count = 0
    for song_id, record in data.items():
        if limit is not None and count >= limit:
            break
        try:
            contents = convert_record(song_id, record)
            output_path = unique_output_path(output_dir, output_filename(song_id, record), used_names)
            output_path.write_text(contents, encoding="utf-8")
            count += 1
        except Exception as exc:  # noqa: BLE001 - report and continue batch conversion.
            errors.append(f"{song_id}: {exc}")

    return count, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Hooktheory JSON.gz records into Humdrum **kern/**mxhm files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="Hooktheory.json.gz",
        type=Path,
        help="Path to Hooktheory.json.gz or decompressed Hooktheory.json.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help=f"Directory for generated .krn files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--limit", type=int, help="Convert only the first N records.")
    args = parser.parse_args()

    count, errors = convert_all(args.input, args.output_dir, args.limit)
    print(f"Converted {count} records into {args.output_dir}")

    if errors:
        error_path = args.output_dir / "conversion_errors.txt"
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Skipped {len(errors)} records; see {error_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
