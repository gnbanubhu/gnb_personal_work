import asyncio


async def iterate_collection(collection):
    for item in collection:
        await asyncio.sleep(0.1)
        print(f"Processing item: {item}")


async def main():
    sample = [10, 20, 30, 40, 50]
    print("Starting async iteration over collection...")
    await iterate_collection(sample)
    print("Async iteration complete.")


if __name__ == "__main__":
    asyncio.run(main())
