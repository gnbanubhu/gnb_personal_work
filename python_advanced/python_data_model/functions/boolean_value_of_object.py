class Basket:

    def __init__(self, items=None):
        self.items = items or []

    def add(self, item):
        self.items.append(item)

    def __bool__(self):
        return len(self.items) > 0

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"Basket({self.items})"


class BankAccount:

    def __init__(self, balance):
        self.balance = balance

    def __bool__(self):
        return self.balance > 0

    def __repr__(self):
        return f"BankAccount(balance={self.balance})"


class Connection:

    def __init__(self, connected):
        self.connected = connected

    def __bool__(self):
        return self.connected

    def __repr__(self):
        return f"Connection(connected={self.connected})"


if __name__ == "__main__":
    print("--- Basket (bool via __bool__ and __len__) ---")
    empty_basket = Basket()
    full_basket = Basket(["Apple", "Banana"])

    print(f"empty_basket    : {empty_basket}")
    print(f"bool(empty)     : {bool(empty_basket)}")
    print(f"full_basket     : {full_basket}")
    print(f"bool(full)      : {bool(full_basket)}")

    if full_basket:
        print("Basket has items — proceed to checkout")
    if not empty_basket:
        print("Basket is empty — nothing to checkout")

    print("\n--- BankAccount (bool via balance) ---")
    active = BankAccount(500)
    overdrawn = BankAccount(-100)
    zero = BankAccount(0)

    print(f"bool(active)    : {bool(active)}")
    print(f"bool(overdrawn) : {bool(overdrawn)}")
    print(f"bool(zero)      : {bool(zero)}")

    print("\n--- Connection (bool via connected flag) ---")
    online = Connection(True)
    offline = Connection(False)

    print(f"bool(online)    : {bool(online)}")
    print(f"bool(offline)   : {bool(offline)}")

    if online:
        print("Connected — sending data")
    if not offline:
        print("Offline — cannot send data")

    print("\n--- Python built-in bool behaviour ---")
    print(f"bool(0)         : {bool(0)}")
    print(f"bool(1)         : {bool(1)}")
    print(f"bool([])        : {bool([])}")
    print(f"bool([1,2])     : {bool([1, 2])}")
    print(f"bool('')        : {bool('')}")
    print(f"bool('hello')   : {bool('hello')}")
    print(f"bool(None)      : {bool(None)}")
