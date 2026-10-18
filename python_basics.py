"""Заготовки задач на базовый Python."""

from math import prod

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    vowels = "aeiou"
    text = data.value.lower()
    # считаем все гласные в строке
    return len([c for c in text if c in vowels])


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    # если в множестве меньше элементов, значит были повторы
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    binary = bin(data.value)[2:]  # убираем префикс 0b
    return binary.count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while len(str(number)) > 1:
        digits = [int(d) for d in str(number)]
        number = prod(digits)
        steps += 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted = data.predicted
    expected = data.expected

    if len(predicted) != len(expected):
        raise ValueError("Векторы должны иметь одинаковую длину.")
    if len(predicted) == 0:
        raise ValueError("Нельзя вычислить MSE для пустых векторов.")

    squared_errors = [(p - e) ** 2 for p, e in zip(predicted, expected)]
    return sum(squared_errors) / len(squared_errors)


def _format_factor(prime: int, power: int) -> str:
    if power == 1:
        return f"({prime})"
    return f"({prime}**{power})"


def prime_factorization(data: PositiveIntegerInput) -> str:
    n = data.value
    result = ""

    # сначала отдельно разбираем двойку, потом идём по нечётным
    divisor = 2
    while divisor * divisor <= n:
        power = 0
        while n % divisor == 0:
            n //= divisor
            power += 1
        if power > 0:
            result += _format_factor(divisor, power)
        divisor += 1 if divisor == 2 else 2

    # остаток больше 1 - это простое число
    if n > 1:
        result += _format_factor(n, 1)

    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cubes = data.value
    height = 0
    total = 0

    # добавляем уровни, пока кубов не хватит
    while total < cubes:
        height += 1
        total += height ** 2

    if total == cubes:
        return height
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = [int(d) for d in str(data.value)]
    n = len(digits)

    # у нечётной длины одна средняя цифра, у чётной две
    half = (n - 1) // 2
    left = digits[:half]
    right = digits[n - half:]

    return sum(left) == sum(right)
