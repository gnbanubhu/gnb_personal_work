import asyncio
from contextlib import contextmanager, asynccontextmanager


class FileManager:

    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        print(f"[__enter__] Opening file: {self.filename}")
        self.file = open(self.filename, "w")
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"[__exit__]  Closing file: {self.filename}")
        if self.file:
            self.file.close()
        if exc_type:
            print(f"[__exit__]  Exception caught: {exc_value}")
        return False


class DatabaseConnection:

    def __init__(self, db_name):
        self.db_name = db_name

    async def __aenter__(self):
        print(f"[__aenter__] Connecting to database: {self.db_name}")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        print(f"[__aexit__]  Disconnecting from database: {self.db_name}")
        await asyncio.sleep(0.1)
        if exc_type:
            print(f"[__aexit__]  Exception caught: {exc_value}")
        return False

    def query(self, sql):
        print(f"Executing query: {sql}")
        return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]


@contextmanager
def timer(label):
    import time
    start = time.time()
    print(f"[timer] Starting: {label}")
    yield
    elapsed = time.time() - start
    print(f"[timer] Finished: {label} in {elapsed:.4f}s")


@asynccontextmanager
async def async_timer(label):
    import time
    start = time.time()
    print(f"[async_timer] Starting: {label}")
    yield
    elapsed = time.time() - start
    print(f"[async_timer] Finished: {label} in {elapsed:.4f}s")


async def main():
    print("--- Class-based Context Manager (with) ---")
    with FileManager("sample.txt") as f:
        f.write("Hello, Context Manager!")
    print()

    print("--- Async Context Manager (async with) ---")
    async with DatabaseConnection("mydb") as db:
        results = db.query("SELECT * FROM users")
        print(f"Query results: {results}")
    print()

    print("--- Generator-based Context Manager (@contextmanager) ---")
    with timer("data processing"):
        await asyncio.sleep(0.2)
    print()

    print("--- Async Generator Context Manager (@asynccontextmanager) ---")
    async with async_timer("async task"):
        await asyncio.sleep(0.2)


if __name__ == "__main__":
    asyncio.run(main())
