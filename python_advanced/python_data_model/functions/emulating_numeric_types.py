import math


class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Arithmetic operators
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    # Comparison operators
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return abs(self) < abs(other)

    def __le__(self, other):
        return abs(self) <= abs(other)

    # Type conversion
    def __bool__(self):
        return bool(self.x or self.y)

    def __int__(self):
        return int(abs(self))

    def __float__(self):
        return float(abs(self))

    # Representation
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __format__(self, format_spec):
        if format_spec == "polar":
            angle = math.atan2(self.y, self.x)
            return f"|{abs(self):.2f}| ∠ {math.degrees(angle):.2f}°"
        return str(self)


if __name__ == "__main__":
    v1 = Vector(3, 4)
    v2 = Vector(1, 2)

    print("--- Arithmetic Operators ---")
    print(f"v1          = {v1}")
    print(f"v2          = {v2}")
    print(f"v1 + v2     = {v1 + v2}")
    print(f"v1 - v2     = {v1 - v2}")
    print(f"v1 * 3      = {v1 * 3}")
    print(f"3 * v1      = {3 * v1}")
    print(f"v1 / 2      = {v1 / 2}")
    print(f"-v1         = {-v1}")

    print("\n--- Absolute Value (magnitude) ---")
    print(f"|v1|        = {abs(v1)}")
    print(f"|v2|        = {abs(v2):.4f}")

    print("\n--- Comparison Operators ---")
    print(f"v1 == v2    : {v1 == v2}")
    print(f"v1 == Vector(3, 4): {v1 == Vector(3, 4)}")
    print(f"v1 < v2     : {v1 < v2}")
    print(f"v1 > v2     : {v1 > v2}")

    print("\n--- Type Conversion ---")
    print(f"bool(v1)    : {bool(v1)}")
    print(f"bool(Vector(0,0)): {bool(Vector(0, 0))}")
    print(f"int(v1)     : {int(v1)}")
    print(f"float(v1)   : {float(v1)}")

    print("\n--- String Representation ---")
    print(f"str(v1)     : {str(v1)}")
    print(f"repr(v1)    : {repr(v1)}")
    print(f"polar format: {v1:polar}")
