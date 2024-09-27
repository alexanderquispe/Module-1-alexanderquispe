"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# A small constant to avoid division by zero or log of zero
epsilon = 1e-6

# Multiplication
def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    return x * y

# Identity function
def id(x: float) -> float:
    "$f(x) = x$"
    return x

# Addition
def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    return x + y

# Negation
def neg(x: float) -> float:
    "$f(x) = -x$"
    return -x

# Less than
def lt(x: float, y: float) -> float:
    "$f(x, y) = 1.0$ if $x < y$, else $0.0$"
    return 1.0 if x < y else 0.0

# Equality check
def eq(x: float, y: float) -> float:
    "$f(x, y) = 1.0$ if $x == y$, else $0.0$"
    return 1.0 if x == y else 0.0

# Maximum of two numbers
def max(x: float, y: float) -> float:
    "$f(x, y) = max(x, y)$"
    return x if x > y else y

# Close to (with tolerance of 1e-2)
def is_close(x: float, y: float) -> float:
    "$f(x, y) = 1.0$ if $|x - y| < 1e-2$, else $0.0$"
    return 1.0 if abs(x - y) < 1e-2 else 0.0

# Sigmoid function
def sigmoid(x: float) -> float:
    "$f(x) = \frac{1.0}{1.0 + e^{-x}}$"
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))

# ReLU (Rectified Linear Unit)
def relu(x: float) -> float:
    "$f(x) = max(0, x)$"
    return max(0, x)

# Natural logarithm with a small epsilon to avoid log(0)
def log(x: float) -> float:
    "$f(x) = log(x + epsilon)$"
    return math.log(x + epsilon)

# Exponential function
def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)

# Logarithm backward (for chain rule in backpropagation)
def log_back(x: float, d: float) -> float:
    r"If $f = log(x)$ as above, compute $d \times f'(x)$"
    return d / (x + epsilon)

# Inverse
def inv(x: float) -> float:
    "$f(x) = \frac{1}{x}$"
    return 1.0 / x

# Inverse backward (for chain rule in backpropagation)
def inv_back(x: float, d: float) -> float:
    r"If $f(x) = \frac{1}{x}$, compute $d \times f'(x)$"
    return -d / (x * x)

# ReLU backward (for chain rule in backpropagation)
def relu_back(x: float, d: float) -> float:
    r"If $f(x) = max(0, x)$, compute $d \times f'(x)$"
    return d if x > 0 else 0


# TODO: Implement for Task 0.1.

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

from functools import reduce as functools_reduce
from typing import Callable, Iterable


# Higher-order function that applies a given function to each element of a list
from typing import Callable, Iterable, List


# Higher-order map function
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], List[float]]:
    """
    Returns a function that applies `fn` to each element of an iterable.

    Args:
        fn: A function that takes a float and returns a float.

    Returns:
        A function that takes an iterable of floats and applies `fn` to each element.
    """

    def apply_to_list(lst: Iterable[float]) -> List[float]:
        return [fn(x) for x in lst]

    return apply_to_list


# Higher-order function that combines elements from two lists using a given function
# Higher-order function that combines two lists using a function, for floats
def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], List[float]]:
    """
    Returns a function that combines elements from two iterables of floats using the function `fn`.

    Args:
        fn: A function that takes two floats and returns a float.

    Returns:
        A function that takes two iterables of floats and returns a list of floats
        by applying `fn` to corresponding elements from both iterables.
    """

    def apply_to_lists(lst1: Iterable[float], lst2: Iterable[float]) -> List[float]:
        return [fn(x, y) for x, y in zip(lst1, lst2)]

    return apply_to_lists

# Higher-order function that reduces a list to a single value using a given function
from typing import Callable, Iterable


# Higher-order function that reduces a list to a single value using a given function
def reduce(fn: Callable[[float, float], float], lst: Iterable[float], start: float) -> float:
    """
    Applies `fn` cumulatively to the items of `lst`, from left to right, to reduce the iterable to a single value.

    Args:
        fn: A function that takes two floats and returns a float.
        lst: An iterable of floats (e.g., list, tuple).
        start: The starting value for the reduction.

    Returns:
        A single float, the result of the reduction.
    """
    result = start
    for item in lst:
        result = fn(result, item)
    return result


# Negate all elements in a list using map
def negList(lst: List[float]) -> List[float]:
    """
    Negate all elements in the list using the higher-order function map_fn.

    Args:
        lst: A list of floats.

    Returns:
        A list of floats where each element is negated.
    """
    return map(neg)(lst)

# Add corresponding elements from two lists using zipWith
def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """
    Add corresponding elements from two lists using the higher-order function zipWith.

    Args:
        lst1: A list of floats.
        lst2: Another list of floats of the same length as lst1.

    Returns:
        A list of floats where each element is the sum of the corresponding elements from lst1 and lst2.
    """
    return zipWith(add)(lst1, lst2)

# Sum all elements in a list using reduce
def sum_list(lst: List[float]) -> float:
    """
    Sum all elements in the list using the higher-order function reduce.

    Args:
        lst: A list of floats.

    Returns:
        The sum of all elements in the list.
    """
    return reduce(add, lst, 0.0)

# Calculate the product of all elements in a list using reduce
def prod(lst: List[float]) -> float:
    """
    Calculate the product of all elements in the list using the higher-order function reduce.

    Args:
        lst: A list of floats.

    Returns:
        The product of all elements in the list.
    """
    return reduce(mul, lst, 1.0)

# TODO: Implement for Task 0.3.
