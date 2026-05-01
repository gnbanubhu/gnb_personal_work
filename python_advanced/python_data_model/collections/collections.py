from collections import namedtuple, deque, OrderedDict, defaultdict, Counter


# namedtuple — immutable record with named fields
def demo_namedtuple():
    print("--- namedtuple ---")
    Point = namedtuple("Point", ["x", "y"])
    p = Point(3, 4)
    print(f"Point       : {p}")
    print(f"x={p.x}, y={p.y}")
    print(f"As dict     : {p._asdict()}")


# deque — double-ended queue, O(1) append/pop from both ends
def demo_deque():
    print("\n--- deque ---")
    dq = deque([1, 2, 3])
    dq.appendleft(0)
    dq.append(4)
    print(f"After appends   : {dq}")
    dq.popleft()
    dq.pop()
    print(f"After pops      : {dq}")
    dq.rotate(1)
    print(f"After rotate(1) : {dq}")


# OrderedDict — dict that remembers insertion order
def demo_ordered_dict():
    print("\n--- OrderedDict ---")
    od = OrderedDict()
    od["banana"] = 3
    od["apple"] = 1
    od["cherry"] = 2
    for key, value in od.items():
        print(f"  {key}: {value}")


# defaultdict — dict with default value for missing keys
def demo_defaultdict():
    print("\n--- defaultdict ---")
    dd = defaultdict(list)
    dd["fruits"].append("Apple")
    dd["fruits"].append("Banana")
    dd["veggies"].append("Carrot")
    print(f"fruits  : {dd['fruits']}")
    print(f"veggies : {dd['veggies']}")
    print(f"missing : {dd['missing']}")


# Counter — counts hashable objects
def demo_counter():
    print("\n--- Counter ---")
    words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
    counter = Counter(words)
    print(f"Counts          : {counter}")
    print(f"Most common(2)  : {counter.most_common(2)}")
    print(f"apple count     : {counter['apple']}")


if __name__ == "__main__":
    demo_namedtuple()
    demo_deque()
    demo_ordered_dict()
    demo_defaultdict()
    demo_counter()
