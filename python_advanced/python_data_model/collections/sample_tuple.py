from collections import namedtuple


def demo_basic_tuple():
    print("--- Basic Tuple ---")
    t = (10, 20, 30, 40, 50)
    print(f"Tuple          : {t}")
    print(f"Access by index: {t[0]}, {t[2]}, {t[4]}")
    print(f"Length         : {len(t)}")
    print(f"Slice [1:3]    : {t[1:3]}")
    print(f"Reversed       : {t[::-1]}")


def demo_tuple_unpacking():
    print("\n--- Tuple Unpacking ---")
    point = (3, 4)
    x, y = point
    print(f"x={x}, y={y}")

    first, *middle, last = (1, 2, 3, 4, 5)
    print(f"first={first}, middle={middle}, last={last}")


def demo_tuple_operations():
    print("\n--- Tuple Operations ---")
    t1 = (1, 2, 3)
    t2 = (4, 5, 6)
    print(f"Concatenation  : {t1 + t2}")
    print(f"Repetition     : {t1 * 3}")
    print(f"Membership     : {3 in t1}")
    print(f"Count of 2     : {t1.count(2)}")
    print(f"Index of 3     : {t1.index(3)}")


def demo_namedtuple():
    print("\n--- Named Tuple ---")
    Employee = namedtuple("Employee", ["name", "department", "salary"])

    emp1 = Employee("Alice", "Engineering", 90000)
    emp2 = Employee("Bob", "Marketing", 75000)

    print(f"emp1           : {emp1}")
    print(f"Name           : {emp1.name}")
    print(f"Department     : {emp1.department}")
    print(f"Salary         : {emp1.salary}")
    print(f"As dict        : {emp1._asdict()}")
    print(f"Fields         : {Employee._fields}")

    updated = emp1._replace(salary=95000)
    print(f"After raise    : {updated}")

    employees = [emp1, emp2]
    print("\nAll Employees:")
    for emp in employees:
        print(f"  {emp.name} | {emp.department} | ${emp.salary:,}")


if __name__ == "__main__":
    demo_basic_tuple()
    demo_tuple_unpacking()
    demo_tuple_operations()
    demo_namedtuple()
