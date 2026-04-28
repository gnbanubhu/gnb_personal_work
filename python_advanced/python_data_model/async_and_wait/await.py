import asyncio


async def fetch_data(item_id):
    print(f"Fetching data for item {item_id}...")
    await asyncio.sleep(1)
    return {"id": item_id, "value": item_id * 10}


async def process_item(item_id):
    data = await fetch_data(item_id)
    print(f"Processed: {data}")
    return data


async def main():
    print("--- Sequential await ---")
    result1 = await process_item(1)
    result2 = await process_item(2)
    result3 = await process_item(3)

    print(f"\nResults: {[result1, result2, result3]}")

    print("\n--- Concurrent await with asyncio.gather ---")
    results = await asyncio.gather(
        process_item(4),
        process_item(5),
        process_item(6),
    )
    print(f"\nResults: {results}")


if __name__ == "__main__":
    asyncio.run(main())
