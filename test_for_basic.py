import unittest
from types import SimpleNamespace

from python_basics import (
    count_vowels,
    has_unique_characters,
    count_one_bits,
    multiplicative_persistence,
    mse,
    prime_factorization,
    pyramid,
    is_balanced_number,
)


# маленькие обёртки, чтобы не писать SimpleNamespace в каждом тесте
def make_text(value):
    return SimpleNamespace(value=value)


def make_number(value):
    return SimpleNamespace(value=value)


def make_vectors(predicted, expected):
    return SimpleNamespace(predicted=predicted, expected=expected)


class TestPythonBasics(unittest.TestCase):

    def test_count_vowels(self):
        self.assertEqual(count_vowels(make_text("hi my name")), 3)
        self.assertEqual(count_vowels(make_text("AEIOU")), 5)    # регистр не важен
        self.assertEqual(count_vowels(make_text("rhythm")), 0)   # гласных нет
        self.assertEqual(count_vowels(make_text("")), 0)         # пустая строка

    def test_unique_characters(self):
        self.assertTrue(has_unique_characters(make_text("abc")))
        self.assertFalse(has_unique_characters(make_text("abca")))
        self.assertFalse(has_unique_characters(make_text("  ")))  # два пробела - повтор
        self.assertTrue(has_unique_characters(make_text("")))

    def test_one_bits(self):
        # (число, сколько единиц в двоичной записи)
        cases = [
            (1, 1),      # 1
            (7, 3),      # 111
            (8, 1),      # 1000
            (255, 8),    # 11111111
        ]
        for number, expected in cases:
            with self.subTest(number=number):
                self.assertEqual(count_one_bits(make_number(number)), expected)

    def test_persistence(self):
        # 39 -> 27 -> 14 -> 4, то есть 3 шага
        self.assertEqual(multiplicative_persistence(make_number(39)), 3)
        # однозначное число, шагов нет
        self.assertEqual(multiplicative_persistence(make_number(4)), 0)
        # 999 -> 729 -> 126 -> 12 -> 2
        self.assertEqual(multiplicative_persistence(make_number(999)), 4)
        # 10 -> 0, один шаг
        self.assertEqual(multiplicative_persistence(make_number(10)), 1)

    def test_mse(self):
        # ошибка только в последнем элементе: 1 / 3
        self.assertAlmostEqual(mse(make_vectors([1, 2, 3], [1, 2, 4])), 1 / 3)
        # (2^2 + 2^2) / 2 = 4
        self.assertEqual(mse(make_vectors([1, 2], [3, 4])), 4.0)
        # одинаковые векторы
        self.assertEqual(mse(make_vectors([5], [5])), 0.0)

    def test_mse_empty_vectors(self):
        # для пустых векторов должна быть ошибка
