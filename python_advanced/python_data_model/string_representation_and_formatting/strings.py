class Product:

    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

    def __str__(self):
        return f"Product: {self.name}, Price: ${self.price:.2f}, Quantity: {self.quantity}"

    def __repr__(self):
        return f"Product(name={self.name!r}, price={self.price}, quantity={self.quantity})"

    def __format__(self, format_spec):
        if format_spec == "summary":
            return f"{self.name} (${self.price:.2f})"
        if format_spec == "full":
            return f"{self.name} | Price: ${self.price:.2f} | Qty: {self.quantity}"
        return str(self)


if __name__ == "__main__":
    product = Product("Laptop", 999.99, 5)

    # __str__: human-readable string
    print("--- __str__ ---")
    print(str(product))
    print(product)

    # __repr__: developer/debug representation
    print("\n--- __repr__ ---")
    print(repr(product))

    # __format__: custom format specs
    print("\n--- __format__ ---")
    print(f"{product:summary}")
    print(f"{product:full}")

    # Built-in string formatting styles
    print("\n--- String Formatting Styles ---")
    name, price = "Laptop", 999.99
    print("%%-style      : Product %s costs $%.2f" % (name, price))
    print("str.format()  : Product {} costs ${:.2f}".format(name, price))
    print(f"f-string      : Product {name} costs ${price:.2f}")

    # Format numbers
    print("\n--- Number Formatting ---")
    pi = 3.14159265
    print(f"Default   : {pi}")
    print(f"2 decimals: {pi:.2f}")
    print(f"Scientific: {pi:.2e}")
    print(f"Percentage: {0.875:.1%}")
    print(f"Thousands : {1000000:,}")
