#!/usr/bin/env python3
"""Evaluate musical conversion fidelity against the Hooktheory JSON source.

This script treats the original Hooktheory annotations as the reference and
parses the generated Humdrum files back into musical events. It measures whether
the conversion preserved melody notes, chord symbols, meters, and keys.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from convert_hooktheory_to_kern import (
    KeyContext,
    chord_symbol,
    key_designation_token,
    key_signature_token,
    output_filename,
    parse_harmony,
    parse_keys,
    parse_melody,
    parse_meters,
    quantize_beat,
    read_json,
    select_context_at_beat,
)


TICKS_PER_BEAT = 960
NATURAL_PITCH_CLASSES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


@dataclass
class Score:
    expected: int
    actual: int
    matched: int

    @property
    def precision(self) -> float:
        return self.matched / self.actual if self.actual else (1.0 if self.expected == 0 else 0.0)

    @property
    def recall(self) -> float:
        return self.matched / self.expected if self.expected else (1.0 if self.actual == 0 else 0.0)

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def as_percent_dict(self) -> dict[str, Any]:
        return {
            "expected": self.expected,
            "actual": self.actual,
            "matched": self.matched,
            "precision_percent": round(self.precision * 100, 6),
            "recall_percent": round(self.recall * 100, 6),
            "f1_percent": round(self.f1 * 100, 6),
        }


@dataclass(frozen=True)
class ParsedKern:
    hooktheory_id: str
    melody: list[tuple[int, int, int, int]]
    chords: list[tuple[int, str]]
    meters: list[tuple[int, int, int]]
    key_signatures: list[tuple[int, str]]
    key_designations: list[tuple[int, str]]


def beat_to_ticks(beat: Fraction) -> int:
    value = beat * TICKS_PER_BEAT
    if value.denominator != 1:
        raise ValueError(f"Beat is outside the {TICKS_PER_BEAT} PPQ grid: {beat}")
    return value.numerator


def duration_to_beats(token: str, beat_unit: int) -> Fraction:
    match = re.match(r"^\[?(\d+)(\.*)", token)
    if not match:
        raise ValueError(f"Missing **kern duration in token: {token}")

    reciprocal = int(match.group(1))
    dots = len(match.group(2))
    dotted_factor = sum(Fraction(1, 2**index) for index in range(dots + 1))
    return Fraction(beat_unit, reciprocal) * dotted_factor


def parse_kern_pitch(token: str) -> tuple[int, int] | None:
    body = re.sub(r"^\[?\d+\.*", "", token)
    body = body.replace("[", "").replace("]", "").replace("_", "")
    body = body.replace("L", "").replace("J", "").replace("K", "").replace("k", "")
    if "r" in body:
        return None

    match = re.match(r"([A-Ga-g]+)([#n-]*)", body)
    if not match:
        raise ValueError(f"Cannot parse **kern pitch token: {token}")

    letters = match.group(1)
    accidentals = match.group(2)
    letter = letters[0].upper()
    if letters[0].islower():
        octave = len(letters) - 1
    else:
        octave = -len(letters)

    accidental = accidentals.count("#") - accidentals.count("-")
    if "n" in accidentals:
        accidental = accidentals.count("#") - accidentals.count("-")

    pitch_class = (NATURAL_PITCH_CLASSES[letter] + accidental) % 12
    return pitch_class, octave


def parse_metadata_id(lines: list[str], path: Path) -> str:
    for line in lines:
        if line.startswith("!!!HTI:"):
            return line.split(":", 1)[1].strip()
    raise ValueError(f"Missing !!!HTI metadata in {path}")


def parse_kern_file(path: Path) -> ParsedKern:
    lines = path.read_text(encoding="utf-8").splitlines()
    song_id = parse_metadata_id(lines, path)
    current_time = Fraction(0)
    beat_unit = 4

    melody: list[tuple[int, int, int, int]] = []
    chords: list[tuple[int, str]] = []
    meters: list[tuple[int, int, int]] = []
    key_signatures: list[tuple[int, str]] = []
    key_designations: list[tuple[int, str]] = []
    active_tie: tuple[Fraction, int, int] | None = None

    for line in lines:
        if not line or line.startswith("!!"):
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            continue

        kern, mxhm = fields
        if kern.startswith("**") or kern == "*-" or kern.startswith("="):
            continue

        meter_match = re.match(r"^\*M(\d+)/(\d+)$", kern)
        if meter_match:
            beats_per_bar = int(meter_match.group(1))
            beat_unit = int(meter_match.group(2))
            meters.append((beat_to_ticks(current_time), beats_per_bar, beat_unit))
            continue

        if kern.startswith("*k["):
            key_signatures.append((beat_to_ticks(current_time), kern))
            continue

        if re.match(r"^\*[A-Ga-g][#n-]*:$", kern):
            key_designations.append((beat_to_ticks(current_time), kern))
            continue

        if kern.startswith("*"):
            continue

        span = duration_to_beats(kern, beat_unit)
        start = current_time
        end = current_time + span

        if mxhm and not mxhm.startswith("*") and not mxhm.startswith("=") and mxhm != ".":
            chords.append((beat_to_ticks(start), mxhm))

        parsed_pitch = parse_kern_pitch(kern)
        if parsed_pitch is not None:
            pitch_class, octave = parsed_pitch
            has_start_tie = "[" in kern
            has_middle_tie = "_" in kern
            has_end_tie = "]" in kern

            if has_start_tie:
                active_tie = (start, pitch_class, octave)
            elif has_middle_tie:
                if active_tie is None:
                    active_tie = (start, pitch_class, octave)
            elif has_end_tie:
                tie_start = start
                if active_tie is not None:
                    tie_start, pitch_class, octave = active_tie
                melody.append((beat_to_ticks(tie_start), beat_to_ticks(end), pitch_class, octave))
                active_tie = None
            else:
                melody.append((beat_to_ticks(start), beat_to_ticks(end), pitch_class, octave))

        current_time = end

    if active_tie is not None:
        tie_start, pitch_class, octave = active_tie
        melody.append((beat_to_ticks(tie_start), beat_to_ticks(current_time), pitch_class, octave))

    return ParsedKern(
        hooktheory_id=song_id,
        melody=melody,
        chords=chords,
        meters=meters,
        key_signatures=key_signatures,
        key_designations=key_designations,
    )


def expected_events(song_id: str, record: dict[str, Any]) -> dict[str, list[tuple[Any, ...]]]:
    annotation = record.get("annotations") or {}
    total_beats = quantize_beat(annotation.get("num_beats", 0))
    keys = parse_keys(annotation)

    melody = [
        (
            beat_to_ticks(event.onset),
            beat_to_ticks(event.offset),
            event.pitch_class,
            event.octave,
        )
        for event in parse_melody(annotation, total_beats)
    ]

    chords = [
        (
            beat_to_ticks(event.onset),
            chord_symbol(event, select_context_at_beat(keys, event.onset)),
        )
        for event in parse_harmony(annotation, total_beats)
    ]

    meters = [
        (
            beat_to_ticks(event.beat),
            event.beats_per_bar,
            event.beat_unit,
        )
        for event in parse_meters(annotation)
    ]

    key_signatures = [
        (
            beat_to_ticks(context.beat),
            key_signature_token(context),
        )
        for context in keys
    ]

    key_designations = [
        (
            beat_to_ticks(context.beat),
            key_designation_token(context),
        )
        for context in keys
    ]

    return {
        "melody": [(song_id, *event) for event in melody],
        "chords": [(song_id, *event) for event in chords],
        "meters": [(song_id, *event) for event in meters],
        "key_signatures": [(song_id, *event) for event in key_signatures],
        "key_designations": [(song_id, *event) for event in key_designations],
    }


def score_events(expected: list[tuple[Any, ...]], actual: list[tuple[Any, ...]]) -> Score:
    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    matched = sum((expected_counter & actual_counter).values())
    return Score(expected=len(expected), actual=len(actual), matched=matched)


def first_mismatches(
    expected: list[tuple[Any, ...]], actual: list[tuple[Any, ...]], limit: int
) -> dict[str, list[tuple[Any, ...]]]:
    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    missing = list((expected_counter - actual_counter).elements())[:limit]
    extra = list((actual_counter - expected_counter).elements())[:limit]
    return {"missing": missing, "extra": extra}


def build_output_id_map(output_dir: Path) -> tuple[dict[str, Path], list[str]]:
    id_map: dict[str, Path] = {}
    duplicate_ids: list[str] = []
    for path in sorted(output_dir.glob("*.krn")):
        lines = path.read_text(encoding="utf-8").splitlines()
        song_id = parse_metadata_id(lines, path)
        if song_id in id_map:
            duplicate_ids.append(song_id)
        id_map[song_id] = path
    return id_map, duplicate_ids


def evaluate(input_path: Path, output_dir: Path, mismatch_limit: int) -> dict[str, Any]:
    data = read_json(input_path)
    output_id_map, duplicate_output_ids = build_output_id_map(output_dir)
    source_ids = set(data)
    output_ids = set(output_id_map)

    missing_output_ids = sorted(source_ids - output_ids)
    extra_output_ids = sorted(output_ids - source_ids)

    expected_by_metric: dict[str, list[tuple[Any, ...]]] = {
        "melody_exact_events": [],
        "chord_onset_symbols": [],
        "meter_events": [],
        "key_signature_events": [],
        "key_designation_events": [],
    }
    actual_by_metric: dict[str, list[tuple[Any, ...]]] = {
        "melody_exact_events": [],
        "chord_onset_symbols": [],
        "meter_events": [],
        "key_signature_events": [],
        "key_designation_events": [],
    }

    parse_errors: list[str] = []

    for song_id, record in data.items():
        expected = expected_events(song_id, record)
        expected_by_metric["melody_exact_events"].extend(expected["melody"])
        expected_by_metric["chord_onset_symbols"].extend(expected["chords"])
        expected_by_metric["meter_events"].extend(expected["meters"])
        expected_by_metric["key_signature_events"].extend(expected["key_signatures"])
        expected_by_metric["key_designation_events"].extend(expected["key_designations"])

        output_path = output_id_map.get(song_id)
        if output_path is None:
            continue

        try:
            parsed = parse_kern_file(output_path)
        except Exception as exc:  # noqa: BLE001 - collect all parse failures.
            parse_errors.append(f"{output_path}: {exc}")
            continue

        actual_by_metric["melody_exact_events"].extend((song_id, *event) for event in parsed.melody)
        actual_by_metric["chord_onset_symbols"].extend((song_id, *event) for event in parsed.chords)
        actual_by_metric["meter_events"].extend((song_id, *event) for event in parsed.meters)
        actual_by_metric["key_signature_events"].extend(
            (song_id, *event) for event in parsed.key_signatures
        )
        actual_by_metric["key_designation_events"].extend(
            (song_id, *event) for event in parsed.key_designations
        )

    scores = {
        metric: score_events(expected_by_metric[metric], actual_by_metric[metric])
        for metric in expected_by_metric
    }

    all_expected: list[tuple[Any, ...]] = []
    all_actual: list[tuple[Any, ...]] = []
    for metric in expected_by_metric:
        all_expected.extend((metric, *event) for event in expected_by_metric[metric])
        all_actual.extend((metric, *event) for event in actual_by_metric[metric])
    overall_score = score_events(all_expected, all_actual)

    mismatches = {
        metric: first_mismatches(
            expected_by_metric[metric], actual_by_metric[metric], mismatch_limit
        )
        for metric in expected_by_metric
        if scores[metric].matched != scores[metric].expected
        or scores[metric].matched != scores[metric].actual
    }

    expected_filenames = Counter(output_filename(song_id, record).casefold() for song_id, record in data.items())
    case_insensitive_filename_collisions = {
        name: count for name, count in expected_filenames.items() if count > 1
    }

    return {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "source_records": len(data),
        "output_files": len(list(output_dir.glob("*.krn"))),
        "missing_output_ids": missing_output_ids,
        "extra_output_ids": extra_output_ids,
        "duplicate_output_ids": duplicate_output_ids,
        "case_insensitive_filename_collision_count": len(case_insensitive_filename_collisions),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:mismatch_limit],
        "metrics": {metric: score.as_percent_dict() for metric, score in scores.items()},
        "overall_event_fidelity": overall_score.as_percent_dict(),
        "mismatches": mismatches,
        "notes": [
            "The Hooktheory JSON annotations are treated as the reference.",
            "Melody exact events compare onset, offset, pitch class, and octave.",
            "Chord accuracy compares chord onset and exported **mxhm symbol; **mxhm does not explicitly encode harmony offsets in these files.",
            "This evaluates conversion fidelity, not whether the original Hooktheory annotations match the commercial recordings.",
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"Source records: {report['source_records']}")
    print(f"Output .krn files: {report['output_files']}")
    print(f"Missing output IDs: {len(report['missing_output_ids'])}")
    print(f"Extra output IDs: {len(report['extra_output_ids'])}")
    print(f"Parse errors: {report['parse_error_count']}")
    print()

    for metric, score in report["metrics"].items():
        print(
            f"{metric}: "
            f"P={score['precision_percent']:.6f}% "
            f"R={score['recall_percent']:.6f}% "
            f"F1={score['f1_percent']:.6f}% "
            f"matched={score['matched']}/{score['expected']}"
        )

    overall = report["overall_event_fidelity"]
    print()
    print(
        "overall_event_fidelity: "
        f"P={overall['precision_percent']:.6f}% "
        f"R={overall['recall_percent']:.6f}% "
        f"F1={overall['f1_percent']:.6f}% "
        f"matched={overall['matched']}/{overall['expected']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate generated .krn files against Hooktheory JSON.")
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
        default="krn",
        type=Path,
        help="Directory containing generated .krn files.",
    )
    parser.add_argument(
        "--report",
        default="accuracy_report.json",
        type=Path,
        help="Where to write the JSON report.",
    )
    parser.add_argument("--mismatch-limit", default=20, type=int)
    args = parser.parse_args()

    report = evaluate(args.input, args.output_dir, args.mismatch_limit)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"\nWrote {args.report}")

    failed = (
        report["missing_output_ids"]
        or report["extra_output_ids"]
        or report["parse_error_count"]
        or any(score["f1_percent"] != 100.0 for score in report["metrics"].values())
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
