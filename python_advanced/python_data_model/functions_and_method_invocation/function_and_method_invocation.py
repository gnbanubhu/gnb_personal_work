class Calculator:

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

    def __call__(self, a, b):
        return a + b


def apply(func, a, b):
    return func(a, b)


def greet(name, message="Hello"):
    return f"{message}, {name}!"


if __name__ == "__main__":
    # Regular function invocation
    print("--- Regular Function Invocation ---")
    print(greet("Alice"))
    print(greet("Bob", message="Welcome"))

    # Method invocation on an instance
    print("\n--- Method Invocation ---")
    calc = Calculator()
    print(f"calc.add(3, 5)      = {calc.add(3, 5)}")
    print(f"calc.multiply(3, 5) = {calc.multiply(3, 5)}")

    # Invoking method via the class (unbound call)
    print("\n--- Unbound Method Invocation ---")
    print(f"Calculator.add(calc, 3, 5) = {Calculator.add(calc, 3, 5)}")

    # Callable object using __call__
    print("\n--- Callable Object via __call__ ---")
    print(f"calc(4, 6) = {calc(4, 6)}")
    print(f"Is callable: {callable(calc)}")

    # Passing a function as an argument (higher-order function)
    print("\n--- Higher-Order Function Invocation ---")
    print(f"apply(calc.add, 10, 20)      = {apply(calc.add, 10, 20)}")
    print(f"apply(calc.multiply, 10, 20) = {apply(calc.multiply, 10, 20)}")

    # Dynamic method invocation using getattr
    print("\n--- Dynamic Method Invocation via getattr ---")
    method = getattr(calc, "add")
    print(f"getattr(calc, 'add')(7, 8) = {method(7, 8)}")
