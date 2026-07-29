import unittest

from molito.core.pt import PT


class TestPeriodicTable(unittest.TestCase):
    def test_symbol_from_atomic(self):
        self.assertEqual(PT.symbol_from_atomic(6), "C")
        self.assertEqual(PT.symbol_from_atomic(7), "N")
        self.assertEqual(PT.symbol_from_atomic(8), "O")
        self.assertEqual(PT.symbol_from_atomic(1), "H")
        self.assertEqual(PT.symbol_from_atomic(16), "S")

    def test_atomic_from_symbol(self):
        self.assertEqual(PT.atomic_from_symbol("C"), 6)
        self.assertEqual(PT.atomic_from_symbol("N"), 7)
        self.assertEqual(PT.atomic_from_symbol("O"), 8)
        self.assertEqual(PT.atomic_from_symbol("H"), 1)

    def test_atomic_from_symbol_case_handling(self):
        # Single-letter symbols get uppercased
        self.assertEqual(PT.atomic_from_symbol("c"), 6)
        self.assertEqual(PT.atomic_from_symbol("n"), 7)

    def test_roundtrip(self):
        for atomic in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]:
            symbol = PT.symbol_from_atomic(atomic)
            result = PT.atomic_from_symbol(symbol)
            self.assertEqual(result, atomic)

    def test_valence(self):
        self.assertEqual(PT.valence(6), 4)
        self.assertEqual(PT.valence(7), 3)
        self.assertEqual(PT.valence(8), 2)
        self.assertEqual(PT.valence(1), 1)

    def test_valence_from_symbol(self):
        self.assertEqual(PT.valence("C"), 4)
        self.assertEqual(PT.valence("N"), 3)
