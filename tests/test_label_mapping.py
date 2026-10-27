import unittest

from label_mapping import LABELS, UNSUPPORTED_LABEL_INDEX, display_label, is_supported_label


class SignMnistLabelMappingTests(unittest.TestCase):
    def test_only_the_24_static_classes_are_supported(self):
        self.assertEqual(len(LABELS), 24)
        self.assertEqual(set(LABELS), set(range(25)) - {UNSUPPORTED_LABEL_INDEX})

    def test_labels_do_not_shift_after_missing_j(self):
        self.assertEqual(LABELS[0], "A")
        self.assertEqual(LABELS[8], "I")
        self.assertEqual(LABELS[10], "K")
        self.assertEqual(LABELS[24], "Y")

    def test_missing_j_index_is_reported_as_unsupported(self):
        self.assertFalse(is_supported_label(UNSUPPORTED_LABEL_INDEX))
        self.assertEqual(display_label(UNSUPPORTED_LABEL_INDEX), "Kelas J tidak tersedia")

    def test_out_of_range_indexes_are_not_mislabelled_as_supported_letters(self):
        self.assertFalse(is_supported_label(-1))
        self.assertFalse(is_supported_label(25))
        self.assertEqual(display_label(-1), "Kelas -1")
        self.assertEqual(display_label(25), "Kelas 25")


if __name__ == "__main__":
    unittest.main()
