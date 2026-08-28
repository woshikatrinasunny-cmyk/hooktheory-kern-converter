import unittest

from convert_hooktheory_to_kern import (
    HarmonyEvent,
    inversion_bass_pitch_class,
)


class InversionTests(unittest.TestCase):
    def test_complete_triad_inversions(self):
        harmony = HarmonyEvent(0, 1, 0, (4, 3), 1)
        self.assertEqual(inversion_bass_pitch_class(harmony), 4)

        harmony = HarmonyEvent(0, 1, 0, (4, 3), 2)
        self.assertEqual(inversion_bass_pitch_class(harmony), 7)

    def test_inversion_survives_omitted_chord_tones(self):
        # Root + fifth can still be marked as a second inversion in the
        # normalized Hooktheory data because the original omit list is gone.
        harmony = HarmonyEvent(0, 1, 0, (7,), 2)
        self.assertEqual(inversion_bass_pitch_class(harmony), 7)

        # Root + fifth + seventh can still be marked as a third inversion
        # when the third was omitted.
        harmony = HarmonyEvent(0, 1, 5, (7, 3), 3)
        self.assertEqual(inversion_bass_pitch_class(harmony), 3)


if __name__ == "__main__":
    unittest.main()
