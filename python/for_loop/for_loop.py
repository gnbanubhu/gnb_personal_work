# ─────────────────────────────────────────────────────────────
# Python For Loop Samples
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 1. Loop over a List
# ─────────────────────────────────────────────────────────────

print("--- Loop over a List ---")
fruits = ["apple", "banana", "cherry", "mango"]
for fruit in fruits:
    print(fruit)


# ─────────────────────────────────────────────────────────────
# 2. Loop over a Range
# ─────────────────────────────────────────────────────────────

print("\n--- Loop over a Range ---")
for i in range(5):          # 0 to 4
    print(i)

print()
for i in range(1, 6):       # 1 to 5
    print(i)

print()
for i in range(0, 11, 2):   # 0 to 10, step 2
    print(i)


# ─────────────────────────────────────────────────────────────
# 3. Loop over a String
# ─────────────────────────────────────────────────────────────

print("\n--- Loop over a String ---")
for char in "Python":
    print(char)


# ─────────────────────────────────────────────────────────────
# 4. Loop over a Tuple
# ─────────────────────────────────────────────────────────────

print("\n--- Loop over a Tuple ---")
coordinates = (10, 20, 30)
for value in coordinates:
    print(value)


# ─────────────────────────────────────────────────────────────
# 5. Loop over a Dictionary
# ─────────────────────────────────────────────────────────────

print("\n--- Loop over a Dictionary ---")
person = {"name": "Alice", "age": 30, "city": "New York"}

for key in person:
    print(key)

print()
for value in person.values():
    print(value)

print()
for key, value in person.items():
    print(f"{key}: {value}")


# ─────────────────────────────────────────────────────────────
# 6. Loop over a Set
# ─────────────────────────────────────────────────────────────

print("\n--- Loop over a Set ---")
unique_ids = {101, 102, 103, 104}
for uid in unique_ids:
    print(uid)


# ─────────────────────────────────────────────────────────────
# 7. enumerate() — Loop with Index
# ─────────────────────────────────────────────────────────────

print("\n--- enumerate() ---")
languages = ["Python", "Java", "Go", "Rust"]
for index, language in enumerate(languages):
    print(f"{index}: {language}")

print()
for index, language in enumerate(languages, start=1):   # start index from 1
    print(f"{index}. {language}")


# ─────────────────────────────────────────────────────────────
# 8. zip() — Loop over Multiple Lists Together
# ─────────────────────────────────────────────────────────────

print("\n--- zip() ---")
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]

for name, score in zip(names, scores):
    print(f"{name}: {score}")


# ─────────────────────────────────────────────────────────────
# 9. Nested For Loop
# ─────────────────────────────────────────────────────────────

print("\n--- Nested For Loop ---")
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i * j}")
    print()


# ─────────────────────────────────────────────────────────────
# 10. break — Exit the Loop Early
# ─────────────────────────────────────────────────────────────

print("--- break ---")
for num in range(1, 10):
    if num == 5:
        print("Stopping at 5")
        break
    print(num)


# ─────────────────────────────────────────────────────────────
# 11. continue — Skip an Iteration
# ─────────────────────────────────────────────────────────────

print("\n--- continue ---")
for num in range(1, 8):
    if num % 2 == 0:    # skip even numbers
        continue
    print(num)


# ─────────────────────────────────────────────────────────────
# 12. else with For Loop
# ─────────────────────────────────────────────────────────────

print("\n--- else with For Loop ---")
for num in range(1, 5):
    print(num)
else:
    print("Loop completed without break")

print()
for num in range(1, 5):
    if num == 3:
        print("Breaking at 3 — else block will NOT run")
        break
    print(num)
else:
    print("This will NOT print because loop was broken")


# ─────────────────────────────────────────────────────────────
# 13. List Comprehension — Compact For Loop
# ─────────────────────────────────────────────────────────────

print("\n--- List Comprehension ---")
squares = [x ** 2 for x in range(1, 6)]
print(squares)                              # [1, 4, 9, 16, 25]

even_squares = [x ** 2 for x in range(1, 11) if x % 2 == 0]
print(even_squares)                         # [4, 16, 36, 64, 100]

upper_fruits = [fruit.upper() for fruit in fruits]
print(upper_fruits)


# ─────────────────────────────────────────────────────────────
# 14. Looping with reversed() and sorted()
# ─────────────────────────────────────────────────────────────

print("\n--- reversed() ---")
for num in reversed(range(1, 6)):
    print(num)

print("\n--- sorted() ---")
unsorted = [3, 1, 4, 1, 5, 9, 2, 6]
for num in sorted(unsorted):
    print(num)

print("\n--- sorted() descending ---")
for num in sorted(unsorted, reverse=True):
    print(num)
