from collections.abc import Iterable

a = {"x": [0, 1, 2], "y": "Lucas"}

print(a)

print(isinstance(a, Iterable))

print(a[0])