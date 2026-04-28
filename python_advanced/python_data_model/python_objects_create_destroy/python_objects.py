class Resource:

    def __new__(cls, name):
        print(f"[__new__]  Allocating memory for '{name}'")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name):
        print(f"[__init__] Initializing object '{name}'")
        self.name = name

    def __del__(self):
        print(f"[__del__]  Destroying object '{self.name}'")

    def display(self):
        print(f"Resource: {self.name}")


if __name__ == "__main__":
    print("--- Creating object ---")
    r1 = Resource("DatabaseConnection")
    r1.display()

    print("\n--- Creating another object ---")
    r2 = Resource("FileHandler")
    r2.display()

    print("\n--- Deleting r1 explicitly ---")
    del r1

    print("\n--- End of program (r2 destroyed automatically) ---")
