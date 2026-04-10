"""
kafka/retail_kafka_app.py
--------------------------
A complete Kafka application for a Retail Event Streaming system.

Demonstrates:
  1. Topic Management    — create, list, describe, delete topics
  2. Producer            — publish retail events (orders, inventory, payments)
  3. Consumer            — consume and process events with group management
  4. Consumer Groups     — multiple consumers sharing partitions
  5. Message Serialization — JSON serialization/deserialization
  6. Partitioning        — key-based partitioning for ordering guarantees
  7. Error Handling      — delivery callbacks, retry logic
  8. Offset Management   — earliest, latest, committed offsets

Retail Events:
  - order_events      : new orders placed by customers
  - inventory_events  : stock updates and reorder alerts
  - payment_events    : payment confirmations and failures

Prerequisites:
  Start Kafka locally before running:
    brew services start zookeeper
    brew services start kafka
  Or:
    zookeeper-server-start /opt/homebrew/etc/kafka/zookeeper.properties
    kafka-server-start /opt/homebrew/etc/kafka/server.properties

Usage:
    python retail_kafka_app.py
"""

import json
import time
import uuid
import random
import threading
from datetime import datetime
from kafka import KafkaAdminClient, KafkaProducer, KafkaConsumer
from kafka.admin import NewTopic
from kafka.errors import TopicAlreadyExistsError, KafkaError

# ── Configuration ──────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = "localhost:9092"
TOPICS = {
    "orders":    {"name": "retail.order.events",     "partitions": 3, "replication": 1},
    "inventory": {"name": "retail.inventory.events", "partitions": 2, "replication": 1},
    "payments":  {"name": "retail.payment.events",   "partitions": 2, "replication": 1},
}
CONSUMER_GROUP = "retail-analytics-group"


def section(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def sub(title):
    print(f"\n  ── {title}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — TOPIC MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 1 — TOPIC MANAGEMENT")

admin = KafkaAdminClient(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    client_id="retail-admin-client"
)

# ── Create topics ─────────────────────────────────────────────────────────────
sub("Creating Topics")

new_topics = [
    NewTopic(
        name=t["name"],
        num_partitions=t["partitions"],
        replication_factor=t["replication"]
    )
    for t in TOPICS.values()
]

for topic in new_topics:
    try:
        admin.create_topics([topic])
        print(f"  Created  : {topic.name} "
              f"(partitions={topic.num_partitions}, "
              f"replication={topic.replication_factor})")
    except TopicAlreadyExistsError:
        print(f"  Exists   : {topic.name} (already created)")

# ── List topics ───────────────────────────────────────────────────────────────
sub("Listing All Topics")
all_topics = sorted(admin.list_topics())
retail_topics = [t for t in all_topics if t.startswith("retail.")]
for t in retail_topics:
    print(f"  {t}")

# ── Describe topics ───────────────────────────────────────────────────────────
sub("Topic Metadata")
metadata = admin.describe_topics([t["name"] for t in TOPICS.values()])
for topic_meta in metadata:
    print(f"\n  Topic     : {topic_meta['topic']}")
    print(f"  Partitions: {len(topic_meta['partitions'])}")
    for p in topic_meta["partitions"]:
        print(f"    partition={p['partition']}  leader={p['leader']}  "
              f"replicas={p['replicas']}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PRODUCER
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 2 — PRODUCER : Publish Retail Events")

# ── Serializers ───────────────────────────────────────────────────────────────
def json_serializer(data):
    return json.dumps(data).encode("utf-8")

def key_serializer(key):
    return str(key).encode("utf-8")

# ── Create Producer ───────────────────────────────────────────────────────────
producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    key_serializer=key_serializer,
    value_serializer=json_serializer,
    acks="all",                  # wait for all replicas to acknowledge
    retries=3,                   # retry on transient failures
    linger_ms=10,                # batch messages for 10ms before sending
    compression_type="gzip",     # compress batches
    client_id="retail-producer"
)

# ── Delivery callback ─────────────────────────────────────────────────────────
delivered = []
failed    = []

def on_send_success(record_metadata, event_type):
    delivered.append(event_type)
    print(f"  ✓ Delivered  topic={record_metadata.topic} "
          f"partition={record_metadata.partition} "
          f"offset={record_metadata.offset}")

def on_send_error(excp, event_type):
    failed.append(event_type)
    print(f"  ✗ Failed     event={event_type}  error={excp}")


# ── Sample data ───────────────────────────────────────────────────────────────
CUSTOMERS  = ["C001", "C002", "C003", "C004", "C005"]
PRODUCTS   = [
    {"id": "P001", "name": "Laptop Pro 15",       "price": 1299.99},
    {"id": "P002", "name": "Wireless Headphones",  "price": 149.99},
    {"id": "P003", "name": "Running Shoes",         "price": 89.99},
    {"id": "P004", "name": "Organic Milk 1L",       "price": 3.49},
    {"id": "P005", "name": "Office Chair",          "price": 299.99},
]
CHANNELS   = ["In-store", "Online", "Mobile"]
PAYMENT_METHODS = ["Card", "Cash", "Online", "Wallet"]


def generate_order_event():
    product  = random.choice(PRODUCTS)
    quantity = random.randint(1, 5)
    return {
        "event_type":   "ORDER_PLACED",
        "order_id":     f"ORD-{uuid.uuid4().hex[:8].upper()}",
        "customer_id":  random.choice(CUSTOMERS),
        "product_id":   product["id"],
        "product_name": product["name"],
        "quantity":     quantity,
        "unit_price":   product["price"],
        "total_amount": round(product["price"] * quantity, 2),
        "channel":      random.choice(CHANNELS),
        "status":       "PENDING",
        "timestamp":    datetime.utcnow().isoformat(),
    }


def generate_inventory_event(product_id):
    stock = random.randint(0, 100)
    return {
        "event_type":      "INVENTORY_UPDATE",
        "product_id":      product_id,
        "quantity_on_hand": stock,
        "reorder_level":   10,
        "is_low_stock":    stock < 10,
        "warehouse":       random.choice(["NYC-WH", "LA-WH", "CHI-WH"]),
        "timestamp":       datetime.utcnow().isoformat(),
    }


def generate_payment_event(order_id, customer_id, amount):
    status = random.choices(
        ["APPROVED", "APPROVED", "APPROVED", "DECLINED", "PENDING"],
        weights=[60, 60, 60, 10, 10]
    )[0]
    return {
        "event_type":      "PAYMENT_PROCESSED",
        "payment_id":      f"PAY-{uuid.uuid4().hex[:8].upper()}",
        "order_id":        order_id,
        "customer_id":     customer_id,
        "amount":          amount,
        "payment_method":  random.choice(PAYMENT_METHODS),
        "status":          status,
        "timestamp":       datetime.utcnow().isoformat(),
    }


# ── Publish order events ──────────────────────────────────────────────────────
sub("Publishing Order Events")
ORDER_COUNT = 8
orders = []

for i in range(ORDER_COUNT):
    event = generate_order_event()
    orders.append(event)
    producer.send(
        topic=TOPICS["orders"]["name"],
        key=event["customer_id"],       # partition by customer_id
        value=event
    ).add_callback(
        lambda meta, e=event["event_type"]: on_send_success(meta, e)
    ).add_errback(
        lambda exc, e=event["event_type"]: on_send_error(exc, e)
    )

producer.flush()
print(f"\n  Orders published : {len(orders)}")

# ── Publish inventory events ──────────────────────────────────────────────────
sub("Publishing Inventory Events")

for product in PRODUCTS:
    event = generate_inventory_event(product["id"])
    producer.send(
        topic=TOPICS["inventory"]["name"],
        key=event["product_id"],        # partition by product_id
        value=event
    ).add_callback(
        lambda meta, e=event["event_type"]: on_send_success(meta, e)
    ).add_errback(
        lambda exc, e=event["event_type"]: on_send_error(exc, e)
    )

producer.flush()
print(f"\n  Inventory events published : {len(PRODUCTS)}")

# ── Publish payment events ────────────────────────────────────────────────────
sub("Publishing Payment Events")

for order in orders[:5]:
    event = generate_payment_event(
        order_id=order["order_id"],
        customer_id=order["customer_id"],
        amount=order["total_amount"]
    )
    producer.send(
        topic=TOPICS["payments"]["name"],
        key=event["order_id"],          # partition by order_id
        value=event
    ).add_callback(
        lambda meta, e=event["event_type"]: on_send_success(meta, e)
    ).add_errback(
        lambda exc, e=event["event_type"]: on_send_error(exc, e)
    )

producer.flush()
print(f"\n  Payment events published : 5")
print(f"\n  Producer Summary — Delivered: {len(delivered)} | Failed: {len(failed)}")

producer.close()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CONSUMER : Consume & Process Events
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 3 — CONSUMER : Consume & Process Events")

def json_deserializer(data):
    return json.loads(data.decode("utf-8")) if data else None


# ── Consumer 1: Orders Consumer ───────────────────────────────────────────────
sub("Consumer 1 — Order Events")

order_consumer = KafkaConsumer(
    TOPICS["orders"]["name"],
    bootstrap_servers=BOOTSTRAP_SERVERS,
    group_id=CONSUMER_GROUP,
    key_deserializer=lambda k: k.decode("utf-8") if k else None,
    value_deserializer=json_deserializer,
    auto_offset_reset="earliest",         # read from beginning
    enable_auto_commit=True,
    auto_commit_interval_ms=1000,
    consumer_timeout_ms=3000,             # stop after 3s of no messages
    client_id="order-consumer-1"
)

order_metrics = {
    "total_orders":   0,
    "total_revenue":  0.0,
    "by_channel":     {},
    "by_customer":    {},
    "by_product":     {},
}

print(f"\n  Consuming from: {TOPICS['orders']['name']}")
print(f"  {'─'*58}")

for msg in order_consumer:
    event = msg.value
    order_metrics["total_orders"]  += 1
    order_metrics["total_revenue"] += event.get("total_amount", 0)

    channel = event.get("channel", "Unknown")
    cust_id = event.get("customer_id", "Unknown")
    prod_id = event.get("product_id",  "Unknown")

    order_metrics["by_channel"][channel]  = order_metrics["by_channel"].get(channel, 0) + 1
    order_metrics["by_customer"][cust_id] = order_metrics["by_customer"].get(cust_id, 0) + 1
    order_metrics["by_product"][prod_id]  = order_metrics["by_product"].get(prod_id,  0) + 1

    print(f"  [partition={msg.partition} offset={msg.offset}] "
          f"order={event.get('order_id')} "
          f"customer={event.get('customer_id')} "
          f"product={event.get('product_name')} "
          f"amount=${event.get('total_amount')}")

order_consumer.close()

print(f"\n  Order Metrics:")
print(f"    Total Orders   : {order_metrics['total_orders']}")
print(f"    Total Revenue  : ${order_metrics['total_revenue']:.2f}")
print(f"    By Channel     : {order_metrics['by_channel']}")
print(f"    By Customer    : {order_metrics['by_customer']}")
print(f"    By Product     : {order_metrics['by_product']}")


# ── Consumer 2: Inventory Consumer ───────────────────────────────────────────
sub("Consumer 2 — Inventory Events")

inventory_consumer = KafkaConsumer(
    TOPICS["inventory"]["name"],
    bootstrap_servers=BOOTSTRAP_SERVERS,
    group_id=CONSUMER_GROUP,
    key_deserializer=lambda k: k.decode("utf-8") if k else None,
    value_deserializer=json_deserializer,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    consumer_timeout_ms=3000,
    client_id="inventory-consumer-1"
)

print(f"\n  Consuming from: {TOPICS['inventory']['name']}")
print(f"  {'─'*58}")

low_stock_alerts = []

for msg in inventory_consumer:
    event = msg.value
    alert = "⚠ LOW STOCK" if event.get("is_low_stock") else "✓ OK"
    print(f"  [partition={msg.partition} offset={msg.offset}] "
          f"product={event.get('product_id')} "
          f"stock={event.get('quantity_on_hand')} "
          f"warehouse={event.get('warehouse')} "
          f"status={alert}")
    if event.get("is_low_stock"):
        low_stock_alerts.append(event.get("product_id"))

inventory_consumer.close()

if low_stock_alerts:
    print(f"\n  Low stock alerts: {low_stock_alerts}")
else:
    print(f"\n  No low stock alerts.")


# ── Consumer 3: Payments Consumer ────────────────────────────────────────────
sub("Consumer 3 — Payment Events")

payment_consumer = KafkaConsumer(
    TOPICS["payments"]["name"],
    bootstrap_servers=BOOTSTRAP_SERVERS,
    group_id=CONSUMER_GROUP,
    key_deserializer=lambda k: k.decode("utf-8") if k else None,
    value_deserializer=json_deserializer,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    consumer_timeout_ms=3000,
    client_id="payment-consumer-1"
)

print(f"\n  Consuming from: {TOPICS['payments']['name']}")
print(f"  {'─'*58}")

payment_summary = {"APPROVED": 0, "DECLINED": 0, "PENDING": 0, "total_amount": 0.0}

for msg in payment_consumer:
    event  = msg.value
    status = event.get("status", "UNKNOWN")
    payment_summary[status]        = payment_summary.get(status, 0) + 1
    payment_summary["total_amount"] += event.get("amount", 0)
    print(f"  [partition={msg.partition} offset={msg.offset}] "
          f"payment={event.get('payment_id')} "
          f"order={event.get('order_id')} "
          f"amount=${event.get('amount')} "
          f"method={event.get('payment_method')} "
          f"status={status}")

payment_consumer.close()

print(f"\n  Payment Summary:")
print(f"    Approved     : {payment_summary['APPROVED']}")
print(f"    Declined     : {payment_summary['DECLINED']}")
print(f"    Pending      : {payment_summary['PENDING']}")
print(f"    Total Amount : ${payment_summary['total_amount']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — CONSUMER GROUP INFO
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 4 — CONSUMER GROUP INFO")

sub("Consumer Group Offsets")
groups = admin.list_consumer_groups()
retail_groups = [g for g in groups if "retail" in g[0]]
print(f"\n  Active consumer groups:")
for g in retail_groups:
    print(f"    group_id={g[0]}  protocol={g[1]}")

offsets = admin.list_consumer_group_offsets(CONSUMER_GROUP)
print(f"\n  Committed offsets for group '{CONSUMER_GROUP}':")
for tp, offset_meta in sorted(offsets.items(), key=lambda x: (x[0].topic, x[0].partition)):
    print(f"    topic={tp.topic:<35} "
          f"partition={tp.partition}  "
          f"offset={offset_meta.offset}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — CLEANUP (optional — comment out to keep topics)
# ══════════════════════════════════════════════════════════════════════════════
section("STAGE 5 — CLEANUP : Delete Topics")

topic_names = [t["name"] for t in TOPICS.values()]
try:
    admin.delete_topics(topic_names)
    print(f"\n  Deleted topics:")
    for t in topic_names:
        print(f"    {t}")
except Exception as e:
    print(f"  Could not delete topics: {e}")

admin.close()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("SUMMARY")
print(f"""
  Component           Description
  ──────────────────────────────────────────────────────────
  Topic Management    Created 3 topics with partitions
  Producer            Published {ORDER_COUNT} orders, {len(PRODUCTS)} inventory, 5 payment events
  Consumers           Consumed all 3 topic streams
  Partitioning        Orders by customer_id, payments by order_id
  Serialization       JSON encode/decode for all messages
  Consumer Group      Offset tracking via '{CONSUMER_GROUP}'
  Error Handling      Delivery callbacks (success + failure)
  Cleanup             Topics deleted after run
  ──────────────────────────────────────────────────────────
  Topics Used:
    retail.order.events      (3 partitions)
    retail.inventory.events  (2 partitions)
    retail.payment.events    (2 partitions)
  Bootstrap Server : {BOOTSTRAP_SERVERS}
""")
