# Hooktheory to Humdrum Kern

This workspace converts Sheet Sage Hooktheory annotations from
`Hooktheory.json.gz` into one Humdrum file per song. Each output uses a
`**kern` melody spine and a `**mxhm` harmony spine, so the files can be loaded in
Verovio Humdrum Viewer.

## Source Data

The converter expects the Hooktheory JSON file from:

https://github.com/chrisdonahue/sheetsage-data/blob/main/hooktheory/Hooktheory.json.gz

The current workspace already contains `Hooktheory.json.gz`.

## Convert

```powershell
python convert_hooktheory_to_kern.py Hooktheory.json.gz -o krn
```

For a quick sample:

```powershell
python convert_hooktheory_to_kern.py Hooktheory.json.gz -o krn_sample --limit 5
```

Generated filenames follow this format:

```text
artist_song_HooktheoryID.krn
```

Example:

```text
green-day_shoplifter_JkmZQMnaoqn.krn
```

On case-insensitive filesystems, if two Hooktheory IDs differ only by case, the
later file receives a suffix such as `__2.krn` so every JSON record still has a
separate output file.

## Validate

```powershell
python validate_kern_outputs.py krn
```

This validation checks Humdrum structure: consistent two-spine rows, the
`**kern`/`**mxhm` exclusive interpretations, and `*-` terminators. Visual
rendering can be checked by opening individual `.krn` files in:

https://verovio.humdrum.org/

## Evaluate Musical Fidelity

```powershell
python evaluate_music_accuracy.py Hooktheory.json.gz -o krn --report accuracy_report.json
```

This treats the original Hooktheory JSON annotations as the reference and parses
the generated `.krn` files back into musical events. The report includes:

- melody exact-event precision/recall/F1 for onset, offset, pitch class, and
  octave;
- chord onset-symbol precision/recall/F1 for the generated `**mxhm` labels;
- meter-event precision/recall/F1;
- key-signature and key-designation precision/recall/F1;
- overall event-fidelity precision/recall/F1.

The current full-run result is 100% overall event fidelity:

```text
overall_event_fidelity: P=100.000000% R=100.000000% F1=100.000000%
matched=1717419/1717419
```

## Conversion Notes

- Metadata is preserved as Humdrum reference records: Hooktheory ID, artist,
  song, Hooktheory URLs, YouTube URL/ID/duration, split, annotators, tags, and
  annotated beat count.
- Beat positions are quantized to a 960 PPQ grid to absorb small floating-point
  offsets in the source JSON.
- Meter and key changes are emitted as Humdrum interpretations at their source
  beat positions.
- Melody gaps are written as rests. Notes that must be split across barlines,
  event boundaries, or uncommon durations are tied with canonical `**kern`
  tie ordering.
- Beam markers are written directly into the `**kern` melody spine using `L`
  and `J`. The converter groups 4/4 melodies by half-measure, compound eighth
  meters such as 6/8 by dotted-quarter beat groups, and other meters by the
  notated beat.
- Harmony events are converted from Hooktheory root-position interval lists into
  MusicXML-style `**mxhm` chord labels such as `C major`, `G dominant`, and
  `D minor-seventh`. Inversions are rendered as slash-bass labels such as
  `D major/F#`; uncommon interval sets use an explicit interval fallback.
