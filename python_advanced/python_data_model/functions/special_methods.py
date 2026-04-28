class ShoppingCart:

    def __init__(self):
        self.items = []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __setitem__(self, index, value):
        self.items[index] = value

    def __delitem__(self, index):
        del self.items[index]

    def __contains__(self, item):
        return item in self.items

    def __iter__(self):
        return iter(self.items)

    def __repr__(self):
        return f"ShoppingCart({self.items})"

    def __str__(self):
        return f"Cart with {len(self.items)} item(s): {self.items}"

    def __add__(self, other):
        merged = ShoppingCart()
        merged.items = self.items + other.items
        return merged

    def __eq__(self, other):
        return self.items == other.items

    def add(self, item):
        self.items.append(item)


if __name__ == "__main__":
    cart1 = ShoppingCart()
    cart1.add("Apple")
    cart1.add("Banana")
    cart1.add("Cherry")

    cart2 = ShoppingCart()
    cart2.add("Mango")
    cart2.add("Orange")

    print("--- __str__ and __repr__ ---")
    print(str(cart1))
    print(repr(cart1))

    print("\n--- __len__ ---")
    print(f"Items in cart1: {len(cart1)}")

    print("\n--- __getitem__ ---")
    print(f"First item: {cart1[0]}")

    print("\n--- __setitem__ ---")
    cart1[0] = "Avocado"
    print(f"After update: {cart1}")

    print("\n--- __delitem__ ---")
    del cart1[0]
    print(f"After delete: {cart1}")

    print("\n--- __contains__ ---")
    print(f"'Banana' in cart1: {'Banana' in cart1}")
    print(f"'Mango' in cart1 : {'Mango' in cart1}")

    print("\n--- __iter__ ---")
    for item in cart1:
        print(f"  Item: {item}")

    print("\n--- __add__ ---")
    merged = cart1 + cart2
    print(f"Merged cart: {merged}")

    print("\n--- __eq__ ---")
    cart3 = ShoppingCart()
    cart3.add("Banana")
    cart3.add("Cherry")
    print(f"cart1 == cart3: {cart1 == cart3}")
    print(f"cart1 == cart1: {cart1 == cart1}")
