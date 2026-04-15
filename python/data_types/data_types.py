# ─────────────────────────────────────────────────────────────
# Python Data Types
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 1. Integer (int)
# ─────────────────────────────────────────────────────────────

age = 25
year = 2024
negative = -10

print("--- Integer ---")
print(type(age))         # <class 'int'>
print(age + year)
print(negative)


# ─────────────────────────────────────────────────────────────
# 2. Float
# ─────────────────────────────────────────────────────────────

price = 9.99
temperature = -3.5
pi = 3.14159

print("\n--- Float ---")
print(type(price))       # <class 'float'>
print(price * 2)
print(round(pi, 2))


# ─────────────────────────────────────────────────────────────
# 3. String (str)
# ─────────────────────────────────────────────────────────────

name = "Alice"
greeting = 'Hello, World!'
multi_line = """This is
a multi-line
string."""

print("\n--- String ---")
print(type(name))               # <class 'str'>
print(name.upper())             # ALICE
print(name.lower())             # alice
print(len(greeting))            # 13
print(greeting.replace("World", "Python"))
print(f"My name is {name}")     # f-string


# ─────────────────────────────────────────────────────────────
# 4. Boolean (bool)
# ─────────────────────────────────────────────────────────────

is_active = True
is_admin = False

print("\n--- Boolean ---")
print(type(is_active))          # <class 'bool'>
print(is_active and is_admin)   # False
print(is_active or is_admin)    # True
print(not is_active)            # False
print(10 > 5)                   # True


# ─────────────────────────────────────────────────────────────
# 5. List
# ─────────────────────────────────────────────────────────────

fruits = ["apple", "banana", "cherry"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]

print("\n--- List ---")
print(type(fruits))             # <class 'list'>
print(fruits[0])                # apple
print(fruits[-1])               # cherry
fruits.append("mango")
print(fruits)
fruits.remove("banana")
print(fruits)
print(len(numbers))             # 5


# ─────────────────────────────────────────────────────────────
# 6. Tuple
# ─────────────────────────────────────────────────────────────

coordinates = (10.0, 20.0)
rgb = (255, 128, 0)
single = (42,)                  # trailing comma required for single-element tuple

print("\n--- Tuple ---")
print(type(coordinates))        # <class 'tuple'>
print(coordinates[0])           # 10.0
print(len(rgb))                 # 3
# Tuples are immutable — coordinates[0] = 5 would raise a TypeError


# ─────────────────────────────────────────────────────────────
# 7. Dictionary (dict)
# ─────────────────────────────────────────────────────────────

person = {
    "name": "Bob",
    "age": 30,
    "city": "New York"
}

print("\n--- Dictionary ---")
print(type(person))             # <class 'dict'>
print(person["name"])           # Bob
print(person.get("age"))        # 30
person["email"] = "bob@example.com"
print(person)
print(person.keys())
print(person.values())


# ─────────────────────────────────────────────────────────────
# 8. Set
# ─────────────────────────────────────────────────────────────

unique_numbers = {1, 2, 3, 3, 4, 4, 5}   # duplicates removed
vowels = {"a", "e", "i", "o", "u"}

print("\n--- Set ---")
print(type(unique_numbers))     # <class 'set'>
print(unique_numbers)           # {1, 2, 3, 4, 5}
unique_numbers.add(6)
unique_numbers.discard(1)
print(unique_numbers)

set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}
print(set_a & set_b)            # Intersection: {3, 4}
print(set_a | set_b)            # Union: {1, 2, 3, 4, 5, 6}
print(set_a - set_b)            # Difference: {1, 2}


# ─────────────────────────────────────────────────────────────
# 9. NoneType
# ─────────────────────────────────────────────────────────────

result = None

print("\n--- NoneType ---")
print(type(result))             # <class 'NoneType'>
print(result is None)           # True
print(result == None)           # True


# ─────────────────────────────────────────────────────────────
# 10. Type Conversion (Casting)
# ─────────────────────────────────────────────────────────────

print("\n--- Type Conversion ---")
print(int(3.9))                 # 3   (truncates, does not round)
print(float(5))                 # 5.0
print(str(100))                 # '100'
print(bool(0))                  # False
print(bool(1))                  # True
print(bool(""))                 # False
print(bool("hello"))            # True
print(list((1, 2, 3)))          # [1, 2, 3]
print(tuple([4, 5, 6]))         # (4, 5, 6)
print(set([1, 1, 2, 3]))        # {1, 2, 3}


# ─────────────────────────────────────────────────────────────
# 11. Checking Data Types
# ─────────────────────────────────────────────────────────────

print("\n--- Checking Types ---")
print(isinstance(42, int))              # True
print(isinstance(3.14, float))          # True
print(isinstance("hello", str))         # True
print(isinstance([1, 2], list))         # True
print(isinstance((1, 2), tuple))        # True
print(isinstance({"a": 1}, dict))       # True
print(isinstance({1, 2}, set))          # True
print(isinstance(True, bool))           # True
print(isinstance(None, type(None)))     # True
