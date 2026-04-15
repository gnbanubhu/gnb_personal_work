# ─────────────────────────────────────────────────────────────
# Python If / Else Statements
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 1. Basic if Statement
# ─────────────────────────────────────────────────────────────

print("--- Basic if ---")
age = 20
if age >= 18:
    print("You are an adult")


# ─────────────────────────────────────────────────────────────
# 2. if / else
# ─────────────────────────────────────────────────────────────

print("\n--- if / else ---")
temperature = 15
if temperature >= 25:
    print("It's warm outside")
else:
    print("It's cold outside")


# ─────────────────────────────────────────────────────────────
# 3. if / elif / else
# ─────────────────────────────────────────────────────────────

print("\n--- if / elif / else ---")
score = 75

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
elif score >= 60:
    print("Grade: D")
else:
    print("Grade: F")


# ─────────────────────────────────────────────────────────────
# 4. Nested if Statements
# ─────────────────────────────────────────────────────────────

print("\n--- Nested if ---")
age = 25
has_id = True

if age >= 18:
    if has_id:
        print("Access granted")
    else:
        print("ID required")
else:
    print("Must be 18 or older")


# ─────────────────────────────────────────────────────────────
# 5. Comparison Operators
# ─────────────────────────────────────────────────────────────

print("\n--- Comparison Operators ---")
x = 10
y = 20

print(x == y)    # False — equal to
print(x != y)    # True  — not equal to
print(x > y)     # False — greater than
print(x < y)     # True  — less than
print(x >= 10)   # True  — greater than or equal to
print(x <= 9)    # False — less than or equal to


# ─────────────────────────────────────────────────────────────
# 6. Logical Operators: and, or, not
# ─────────────────────────────────────────────────────────────

print("\n--- Logical Operators ---")
age = 22
has_ticket = True
is_vip = False

# and — both conditions must be True
if age >= 18 and has_ticket:
    print("Entry allowed")

# or — at least one condition must be True
if is_vip or has_ticket:
    print("Can enter the event")

# not — reverses the condition
if not is_vip:
    print("Not a VIP member")


# ─────────────────────────────────────────────────────────────
# 7. Membership Operators: in, not in
# ─────────────────────────────────────────────────────────────

print("\n--- Membership Operators ---")
fruits = ["apple", "banana", "cherry"]

if "banana" in fruits:
    print("Banana is in the list")

if "mango" not in fruits:
    print("Mango is not in the list")

username = "admin"
allowed_users = ["admin", "superuser", "editor"]

if username in allowed_users:
    print(f"{username} has access")


# ─────────────────────────────────────────────────────────────
# 8. Identity Operators: is, is not
# ─────────────────────────────────────────────────────────────

print("\n--- Identity Operators ---")
result = None

if result is None:
    print("No result returned")

value = 42
if value is not None:
    print(f"Value is: {value}")


# ─────────────────────────────────────────────────────────────
# 9. Ternary Operator (One-line if/else)
# ─────────────────────────────────────────────────────────────

print("\n--- Ternary Operator ---")
age = 20
status = "adult" if age >= 18 else "minor"
print(f"Status: {status}")

num = 7
parity = "even" if num % 2 == 0 else "odd"
print(f"{num} is {parity}")


# ─────────────────────────────────────────────────────────────
# 10. Truthy and Falsy Values
# ─────────────────────────────────────────────────────────────

print("\n--- Truthy and Falsy ---")

# Falsy values: 0, "", [], {}, (), None, False
falsy_values = [0, "", [], {}, (), None, False]
for val in falsy_values:
    if not val:
        print(f"{repr(val)} is Falsy")

print()
# Truthy values: non-zero, non-empty
truthy_values = [1, "hello", [1, 2], {"a": 1}, (1,), True]
for val in truthy_values:
    if val:
        print(f"{repr(val)} is Truthy")


# ─────────────────────────────────────────────────────────────
# 11. Chained Comparisons
# ─────────────────────────────────────────────────────────────

print("\n--- Chained Comparisons ---")
x = 15

if 10 < x < 20:
    print(f"{x} is between 10 and 20")

age = 25
if 18 <= age <= 60:
    print(f"Age {age} is in working range")


# ─────────────────────────────────────────────────────────────
# 12. if with Functions
# ─────────────────────────────────────────────────────────────

print("\n--- if with Functions ---")

def classify_number(n):
    if n > 0:
        return "Positive"
    elif n < 0:
        return "Negative"
    else:
        return "Zero"

for num in [-5, 0, 10]:
    print(f"{num}: {classify_number(num)}")


# ─────────────────────────────────────────────────────────────
# 13. if with String Conditions
# ─────────────────────────────────────────────────────────────

print("\n--- if with Strings ---")
name = "Alice"

if name.startswith("A"):
    print(f"{name} starts with A")

if name.endswith("e"):
    print(f"{name} ends with e")

if "lic" in name:
    print(f"'lic' found in {name}")

if name.lower() == "alice":
    print("Name matches (case-insensitive)")


# ─────────────────────────────────────────────────────────────
# 14. match / case (Python 3.10+ — Structural Pattern Matching)
# ─────────────────────────────────────────────────────────────

print("\n--- match / case (Python 3.10+) ---")

def http_status(status_code):
    match status_code:
        case 200:
            return "OK"
        case 201:
            return "Created"
        case 400:
            return "Bad Request"
        case 401:
            return "Unauthorized"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:
            return "Unknown Status"

for code in [200, 201, 404, 500, 999]:
    print(f"{code}: {http_status(code)}")
