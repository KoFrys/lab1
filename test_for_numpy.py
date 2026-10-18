import unittest
from types import SimpleNamespace as NS

import numpy as np

from numpy_tasks import (
    sum_prod,
    binarize,
    unique_rows,
    unique_columns,
    matrix_statistics,
    chess,
    draw_rectangle,
    draw_ellipse,
    analyze_time_series,
    one_hot,
)


class TestNumpyTasks(unittest.TestCase):

    # ---------- Задача 1. sum_prod ----------
    def test_sum_prod(self):
        data = NS(
            matrices=[np.array([[1, 0], [0, 2]]), np.array([[2, 1], [1, 1]])],
            vectors=[np.array([[3], [4]]), np.array([[1], [2]])],
        )

        # первое произведение: [[3], [8]]
        # второе произведение: [[4], [3]]
        # сумма: [[7], [11]]
        result = sum_prod(data)

        np.testing.assert_array_equal(result, [[7], [11]])
        self.assertEqual(result.shape, (2, 1))

    def test_sum_prod_single_pair(self):
        # единичная матрица не должна менять вектор
        data = NS(matrices=[np.eye(2)], vectors=[np.array([[7], [8]])])

        result = sum_prod(data)

        np.testing.assert_array_equal(result, [[7], [8]])

    # ---------- Задача 2. binarize ----------
    def test_binarize(self):
        matrix = np.array([[1, 2], [3, 2]])

        result = binarize(NS(matrix=matrix, threshold=2))

        # строго больше порога -> 1, иначе 0
        np.testing.assert_array_equal(result, [[0, 0], [1, 0]])

        # исходную матрицу трогать нельзя
        np.testing.assert_array_equal(matrix, [[1, 2], [3, 2]])
        self.assertIsNot(result, matrix)

    def test_binarize_negative_threshold(self):
        data = NS(matrix=np.array([[-3, -1, 0, 2]]), threshold=-1)

        result = binarize(data)

        np.testing.assert_array_equal(result, [[0, 0, 1, 1]])

    # ---------- Задача 3. unique_rows / unique_columns ----------
    def test_unique_rows(self):
        matrix = np.array([
            [2, 1, 2],
            [4, 4, 3],
            [5, 5, 5],
        ])

        result = unique_rows(NS(matrix=matrix))

        self.assertEqual(result, [[1, 2], [3, 4], [5]])

    def test_unique_columns(self):
        matrix = np.array([
            [2, 1, 2],
            [4, 1, 3],
            [2, 5, 3],
        ])

        result = unique_columns(NS(matrix=matrix))

        self.assertEqual(result, [[2, 4], [1, 5], [2, 3]])

    # ---------- Задача 4. matrix_statistics ----------
    def test_matrix_statistics(self):
        data = NS(rows=3, columns=4, mean=2.0, std=0.5, seed=123)

        result = matrix_statistics(data)
        matrix = result.matrix

        # проверяем размер
        self.assertEqual(matrix.shape, (3, 4))

        # сравниваем статистики с тем, что считает сам numpy
        np.testing.assert_allclose(result.row_means, matrix.mean(axis=1))
        np.testing.assert_allclose(result.column_means, matrix.mean(axis=0))
        np.testing.assert_allclose(result.row_variances, matrix.var(axis=1))
        np.testing.assert_allclose(result.column_variances, matrix.var(axis=0))

    def test_matrix_statistics_reproducible(self):
        data = NS(rows=3, columns=3, mean=0.0, std=1.0, seed=42)

        # с одинаковым seed матрицы должны совпасть
        first = matrix_statistics(data)
        second = matrix_statistics(data)

        np.testing.assert_array_equal(first.matrix, second.matrix)

    # ---------- Задача 5. chess ----------
    def test_chess(self):
        result = chess(NS(rows=3, columns=4, first=0, second=1))

        expected = [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
        np.testing.assert_array_equal(result, expected)

    def test_chess_one_element(self):
        result = chess(NS(rows=1, columns=1, first=8, second=9))

        # в левом верхнем углу всегда первое значение
        np.testing.assert_array_equal(result, [[8]])

    # ---------- Задача 6. draw_rectangle / draw_ellipse ----------
    def test_draw_rectangle(self):
        red = [255, 0, 0]
        black = [0, 0, 0]
        data = NS(
            width=2,
            height=2,
            image_height=5,
            image_width=5,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )

        image = draw_rectangle(data)

        self.assertEqual(image.shape, (5, 5, 3))
        np.testing.assert_array_equal(image[0, 0], black)  # угол - фон
        np.testing.assert_array_equal(image[2, 2], red)    # центр - фигура

        # у прямоугольника 2x2 должно быть ровно 4 цветных пикселя
        red_mask = np.all(image == red, axis=2)
        self.assertEqual(int(red_mask.sum()), 4)

    def test_draw_ellipse(self):
        data = NS(
            semi_axis_x=2,
            semi_axis_y=1,
            image_height=5,
            image_width=5,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )

        image = draw_ellipse(data)

        self.assertEqual(image.shape, (5, 5, 3))
        np.testing.assert_array_equal(image[2, 2], [255, 0, 0])  # центр
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])    # угол

    # ---------- Задача 7. analyze_time_series ----------
    def test_analyze_time_series(self):
        values = [1, 3, 1, 0, 2, 0]

        result = analyze_time_series(NS(values=values, window=2))

        self.assertAlmostEqual(result.mean, np.mean(values))
        self.assertAlmostEqual(result.variance, np.var(values))
        self.assertAlmostEqual(result.std, np.std(values))

        # максимумы на позициях 1 и 4, минимум на позиции 3
        self.assertEqual(result.local_maxima_indices, [1, 4])
        self.assertEqual(result.local_minima_indices, [3])

        np.testing.assert_allclose(result.moving_average, [2, 2, 0.5, 1, 1])

    def test_time_series_window_one(self):
        result = analyze_time_series(NS(values=[1, 2, 3], window=1))

        # окно 1 - ряд не меняется
        np.testing.assert_allclose(result.moving_average, [1, 2, 3])

    # ---------- Задача 8. one_hot ----------
    def test_one_hot(self):
        data = NS(labels=[0, 2, 3, 0], class_count=None)

        result = one_hot(data)

        expected = [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
        np.testing.assert_array_equal(result, expected)

    def test_one_hot_explicit_class_count(self):
        data = NS(labels=[1, 0], class_count=3)

        result = one_hot(data)

        np.testing.assert_array_equal(result, [[0, 1, 0], [1, 0, 0]])

    def test_one_hot_empty(self):
        result = one_hot(NS(labels=[], class_count=None))

        self.assertEqual(result.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
