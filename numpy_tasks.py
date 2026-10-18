"""Заготовки задач на NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)

    if len(matrices) != len(vectors):
        raise ValueError("Количество матриц и векторов должно совпадать.")
    if len(matrices) == 0:
        raise ValueError("Набор матриц и векторов не должен быть пустым.")

    # перемножаем сразу все пары: (p, n, n) @ (p, n, 1) -> (p, n, 1)
    products = matrices @ vectors

    # потом складываем по первой оси
    return np.sum(products, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = np.asarray(data.matrix)
    # где больше порога - 1, иначе 0
    return np.where(matrix > data.threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for row in matrix:
        result.append(np.unique(row).tolist())
    return result


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    result = []
    for j in range(matrix.shape[1]):
        result.append(np.unique(matrix[:, j]).tolist())
    return result


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(loc=data.mean, scale=data.std, size=(data.rows, data.columns))

    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    # i и j - индексы строки и столбца каждой клетки
    i, j = np.indices((data.rows, data.columns))

    # чётная сумма индексов - первый цвет, нечётная - второй
    return np.where((i + j) % 2 == 0, data.first, data.second)


def _make_canvas(height: int, width: int, color) -> np.ndarray:
    """Создаёт RGB-картинку, залитую одним цветом."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width = data.width
    height = data.height

    if width < 0 or height < 0:
        raise ValueError("Ширина и высота должны быть неотрицательными.")

    image = _make_canvas(data.image_height, data.image_width, data.background_color)

    # левый верхний угол прямоугольника
    x_start = (data.image_width - width) // 2
    y_start = (data.image_height - height) // 2

    # обрезаем по границам картинки, если прямоугольник не помещается
    left = max(0, x_start)
    right = min(data.image_width, x_start + width)
    top = max(0, y_start)
    bottom = min(data.image_height, y_start + height)

    if left < right and top < bottom:
        image[top:bottom, left:right] = data.shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    a = data.semi_axis_x
    b = data.semi_axis_y

    if a <= 0 or b <= 0:
        raise ValueError("Полуоси эллипса должны быть положительными.")

    image = _make_canvas(data.image_height, data.image_width, data.background_color)

    # центр картинки
    center_x = (data.image_width - 1) / 2
    center_y = (data.image_height - 1) / 2

    y, x = np.ogrid[:data.image_height, :data.image_width]

    # точка внутри эллипса, если (x/a)^2 + (y/b)^2 <= 1
    mask = ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1

    image[mask] = data.shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)
    window = data.window

    if values.ndim != 1:
        raise ValueError("Временной ряд должен быть одномерным.")
    if window < 1 or window > len(values):
        raise ValueError("Размер окна должен быть от 1 до длины временного ряда.")

    maxima = []
    minima = []

    # крайние точки пропускаем, у них нет двух соседей
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            maxima.append(i)
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            minima.append(i)

    # скользящее среднее через свёртку с окном из 1/window
    kernel = np.ones(window) / window
    moving_average = np.convolve(values, kernel, mode="valid")

    return TimeSeriesStatistics(
        mean=float(values.mean()),
        variance=float(values.var()),
        std=float(values.std()),
        local_maxima_indices=maxima,
        local_minima_indices=minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    class_count = data.class_count

    if labels.ndim != 1:
        raise ValueError("Метки должны быть одномерным вектором.")
    if np.any(labels < 0):
        raise ValueError("Метки классов должны быть неотрицательными.")

    # если число классов не дали, берём максимальную метку + 1
    if class_count is None:
        if labels.size > 0:
            class_count = int(labels.max()) + 1
        else:
            class_count = 0

    if class_count < 0:
        raise ValueError("Количество классов не может быть отрицательным.")
    if labels.size > 0 and np.any(labels >= class_count):
        raise ValueError("Метка выходит за пределы class_count.")

    result = np.zeros((len(labels), class_count), dtype=int)

    # в каждой строке ставим единицу в столбец, равный метке
    if labels.size > 0:
        result[np.arange(len(labels)), labels] = 1

    return result


def plot_matrix_statistics(statistics: MatrixStatistics) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(statistics.matrix)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # слева средние по строкам, справа по столбцам
    axes[0].hist(matrix.mean(axis=1), bins="auto")
    axes[0].set_title("Средние значения строк")
    axes[0].set_xlabel("Среднее")
    axes[0].set_ylabel("Частота")

    axes[1].hist(matrix.mean(axis=0), bins="auto")
    axes[1].set_title("Средние значения столбцов")
    axes[1].set_xlabel("Среднее")
    axes[1].set_ylabel("Частота")

    fig.tight_layout()
    plt.show()
