"""Schema definitions across domains.

15 training schemas + 3 held-out schemas (never in training set).
"""
from __future__ import annotations

from text_to_mongo.schema import FieldDef, FieldRole, SchemaDef

# ---------------------------------------------------------------------------
# E-commerce
# ---------------------------------------------------------------------------
ORDERS = SchemaDef(
    collection="orders",
    domain="ecommerce",
    fields=[
        FieldDef(name="order_id", type="string", role=FieldRole.identifier, description="Unique order identifier"),
        FieldDef(name="customer_id", type="string", role=FieldRole.identifier, description="Customer reference"),
        FieldDef(name="total_amount", type="double", role=FieldRole.measure, description="Order total in USD"),
        FieldDef(name="order_date", type="date", role=FieldRole.timestamp, description="Date the order was placed"),
        FieldDef(name="status", type="string", role=FieldRole.enum, description="Order status",
                 enum_values=["pending", "shipped", "delivered", "cancelled"]),
        FieldDef(name="channel", type="string", role=FieldRole.category, description="Sales channel"),
    ],
)

PRODUCTS = SchemaDef(
    collection="products",
    domain="ecommerce",
    fields=[
        FieldDef(name="product_id", type="string", role=FieldRole.identifier, description="Product SKU"),
        FieldDef(name="name", type="string", role=FieldRole.text, description="Product name"),
        FieldDef(name="price", type="double", role=FieldRole.measure, description="Unit price"),
        FieldDef(name="category", type="string", role=FieldRole.category, description="Product category"),
        FieldDef(name="in_stock", type="bool", role=FieldRole.boolean, description="Whether product is in stock"),
        FieldDef(name="rating", type="double", role=FieldRole.measure, description="Average customer rating"),
        FieldDef(name="created_at", type="date", role=FieldRole.timestamp, description="Date product was listed"),
    ],
)

CUSTOMERS = SchemaDef(
    collection="customers",
    domain="ecommerce",
    fields=[
        FieldDef(name="customer_id", type="string", role=FieldRole.identifier, description="Unique customer ID"),
        FieldDef(name="email", type="string", role=FieldRole.text, description="Email address"),
        FieldDef(name="lifetime_value", type="double", role=FieldRole.measure, description="Total spend to date"),
        FieldDef(name="signup_date", type="date", role=FieldRole.timestamp, description="Account creation date"),
        FieldDef(name="tier", type="string", role=FieldRole.enum, description="Loyalty tier",
                 enum_values=["bronze", "silver", "gold", "platinum"]),
        FieldDef(name="region", type="string", role=FieldRole.category, description="Geographic region"),
    ],
)

# ---------------------------------------------------------------------------
# Healthcare
# ---------------------------------------------------------------------------
PATIENT_VISITS = SchemaDef(
    collection="patient_visits",
    domain="healthcare",
    fields=[
        FieldDef(name="visit_id", type="string", role=FieldRole.identifier, description="Visit identifier"),
        FieldDef(name="patient_id", type="string", role=FieldRole.identifier, description="Patient reference"),
        FieldDef(name="charge", type="double", role=FieldRole.measure, description="Visit charge in USD"),
        FieldDef(name="visit_date", type="date", role=FieldRole.timestamp, description="Date of visit"),
        FieldDef(name="department", type="string", role=FieldRole.category, description="Hospital department"),
        FieldDef(name="diagnosis", type="string", role=FieldRole.text, description="Primary diagnosis"),
        FieldDef(name="is_emergency", type="bool", role=FieldRole.boolean, description="Whether visit was emergency"),
    ],
)

LAB_RESULTS = SchemaDef(
    collection="lab_results",
    domain="healthcare",
    fields=[
        FieldDef(name="result_id", type="string", role=FieldRole.identifier, description="Result identifier"),
        FieldDef(name="patient_id", type="string", role=FieldRole.identifier, description="Patient reference"),
        FieldDef(name="value", type="double", role=FieldRole.measure, description="Test result value"),
        FieldDef(name="collected_at", type="date", role=FieldRole.timestamp, description="Sample collection date"),
        FieldDef(name="test_type", type="string", role=FieldRole.category, description="Type of lab test"),
        FieldDef(name="status", type="string", role=FieldRole.enum, description="Result status",
                 enum_values=["pending", "completed", "reviewed"]),
    ],
)

# ---------------------------------------------------------------------------
# IoT
# ---------------------------------------------------------------------------
SENSOR_READINGS = SchemaDef(
    collection="sensor_readings",
    domain="iot",
    fields=[
        FieldDef(name="sensor_id", type="string", role=FieldRole.identifier, description="Sensor identifier"),
        FieldDef(name="reading", type="double", role=FieldRole.measure, description="Sensor reading value"),
        FieldDef(name="timestamp", type="date", role=FieldRole.timestamp, description="Reading timestamp"),
        FieldDef(name="location", type="string", role=FieldRole.category, description="Sensor location"),
        FieldDef(name="unit", type="string", role=FieldRole.enum, description="Measurement unit",
                 enum_values=["celsius", "fahrenheit", "psi", "rpm", "kwh"]),
        FieldDef(name="is_anomaly", type="bool", role=FieldRole.boolean, description="Whether reading is anomalous"),
    ],
)

DEVICE_LOGS = SchemaDef(
    collection="device_logs",
    domain="iot",
    fields=[
        FieldDef(name="log_id", type="string", role=FieldRole.identifier, description="Log entry ID"),
        FieldDef(name="device_id", type="string", role=FieldRole.identifier, description="Device reference"),
        FieldDef(name="severity", type="string", role=FieldRole.enum, description="Log severity",
                 enum_values=["debug", "info", "warning", "error", "critical"]),
        FieldDef(name="logged_at", type="date", role=FieldRole.timestamp, description="Log timestamp"),
        FieldDef(name="message", type="string", role=FieldRole.text, description="Log message"),
        FieldDef(name="response_time_ms", type="int", role=FieldRole.measure, description="Response time in ms"),
    ],
)

# ---------------------------------------------------------------------------
# HR
# ---------------------------------------------------------------------------
EMPLOYEES = SchemaDef(
    collection="employees",
    domain="hr",
    fields=[
        FieldDef(name="employee_id", type="string", role=FieldRole.identifier, description="Employee ID"),
        FieldDef(name="name", type="string", role=FieldRole.text, description="Full name"),
        FieldDef(name="salary", type="double", role=FieldRole.measure, description="Annual salary"),
        FieldDef(name="hire_date", type="date", role=FieldRole.timestamp, description="Date of hire"),
        FieldDef(name="department", type="string", role=FieldRole.category, description="Department name"),
        FieldDef(name="level", type="string", role=FieldRole.enum, description="Job level",
                 enum_values=["junior", "mid", "senior", "lead", "director"]),
        FieldDef(name="is_active", type="bool", role=FieldRole.boolean, description="Currently employed"),
    ],
)

PERFORMANCE_REVIEWS = SchemaDef(
    collection="performance_reviews",
    domain="hr",
    fields=[
        FieldDef(name="review_id", type="string", role=FieldRole.identifier, description="Review ID"),
        FieldDef(name="employee_id", type="string", role=FieldRole.identifier, description="Employee reference"),
        FieldDef(name="score", type="double", role=FieldRole.measure, description="Performance score (1-5)"),
        FieldDef(name="review_date", type="date", role=FieldRole.timestamp, description="Review date"),
        FieldDef(name="reviewer", type="string", role=FieldRole.text, description="Reviewer name"),
        FieldDef(name="category", type="string", role=FieldRole.category, description="Review category"),
    ],
)

# ---------------------------------------------------------------------------
# Finance
# ---------------------------------------------------------------------------
TRANSACTIONS = SchemaDef(
    collection="transactions",
    domain="finance",
    fields=[
        FieldDef(name="txn_id", type="string", role=FieldRole.identifier, description="Transaction ID"),
        FieldDef(name="account_id", type="string", role=FieldRole.identifier, description="Account reference"),
        FieldDef(name="amount", type="double", role=FieldRole.measure, description="Transaction amount"),
        FieldDef(name="txn_date", type="date", role=FieldRole.timestamp, description="Transaction date"),
        FieldDef(name="type", type="string", role=FieldRole.enum, description="Transaction type",
                 enum_values=["debit", "credit", "transfer", "fee"]),
        FieldDef(name="merchant", type="string", role=FieldRole.category, description="Merchant name"),
        FieldDef(name="is_flagged", type="bool", role=FieldRole.boolean, description="Flagged for review"),
    ],
)

ACCOUNTS = SchemaDef(
    collection="accounts",
    domain="finance",
    fields=[
        FieldDef(name="account_id", type="string", role=FieldRole.identifier, description="Account ID"),
        FieldDef(name="owner_name", type="string", role=FieldRole.text, description="Account owner name"),
        FieldDef(name="balance", type="double", role=FieldRole.measure, description="Current balance"),
        FieldDef(name="opened_at", type="date", role=FieldRole.timestamp, description="Account opening date"),
        FieldDef(name="account_type", type="string", role=FieldRole.enum, description="Account type",
                 enum_values=["checking", "savings", "credit", "investment"]),
        FieldDef(name="branch", type="string", role=FieldRole.category, description="Branch name"),
    ],
)

# ---------------------------------------------------------------------------
# Logistics
# ---------------------------------------------------------------------------
SHIPMENTS = SchemaDef(
    collection="shipments",
    domain="logistics",
    fields=[
        FieldDef(name="shipment_id", type="string", role=FieldRole.identifier, description="Shipment ID"),
        FieldDef(name="weight_kg", type="double", role=FieldRole.measure, description="Shipment weight in kg"),
        FieldDef(name="shipped_at", type="date", role=FieldRole.timestamp, description="Shipment date"),
        FieldDef(name="carrier", type="string", role=FieldRole.category, description="Shipping carrier"),
        FieldDef(name="status", type="string", role=FieldRole.enum, description="Shipment status",
                 enum_values=["processing", "in_transit", "delivered", "returned"]),
        FieldDef(name="is_fragile", type="bool", role=FieldRole.boolean, description="Fragile handling required"),
    ],
)

WAREHOUSES = SchemaDef(
    collection="warehouses",
    domain="logistics",
    fields=[
        FieldDef(name="warehouse_id", type="string", role=FieldRole.identifier, description="Warehouse ID"),
        FieldDef(name="capacity", type="int", role=FieldRole.measure, description="Max storage units"),
        FieldDef(name="opened_date", type="date", role=FieldRole.timestamp, description="Date warehouse opened"),
        FieldDef(name="region", type="string", role=FieldRole.category, description="Geographic region"),
        FieldDef(name="is_climate_controlled", type="bool", role=FieldRole.boolean, description="Has climate control"),
    ],
)

# ---------------------------------------------------------------------------
# Social
# ---------------------------------------------------------------------------
POSTS = SchemaDef(
    collection="posts",
    domain="social",
    fields=[
        FieldDef(name="post_id", type="string", role=FieldRole.identifier, description="Post ID"),
        FieldDef(name="author_id", type="string", role=FieldRole.identifier, description="Author reference"),
        FieldDef(name="likes", type="int", role=FieldRole.measure, description="Number of likes"),
        FieldDef(name="published_at", type="date", role=FieldRole.timestamp, description="Publication timestamp"),
        FieldDef(name="topic", type="string", role=FieldRole.category, description="Post topic"),
        FieldDef(name="content", type="string", role=FieldRole.text, description="Post body text"),
        FieldDef(name="is_pinned", type="bool", role=FieldRole.boolean, description="Whether post is pinned"),
    ],
)

USER_ACTIVITY = SchemaDef(
    collection="user_activity",
    domain="social",
    fields=[
        FieldDef(name="activity_id", type="string", role=FieldRole.identifier, description="Activity ID"),
        FieldDef(name="user_id", type="string", role=FieldRole.identifier, description="User reference"),
        FieldDef(name="duration_sec", type="int", role=FieldRole.measure, description="Activity duration in seconds"),
        FieldDef(name="occurred_at", type="date", role=FieldRole.timestamp, description="Activity timestamp"),
        FieldDef(name="action_type", type="string", role=FieldRole.enum, description="Type of action",
                 enum_values=["login", "logout", "post", "comment", "share"]),
        FieldDef(name="platform", type="string", role=FieldRole.category, description="Client platform"),
    ],
)

# ---------------------------------------------------------------------------
# Education
# ---------------------------------------------------------------------------
ENROLLMENTS = SchemaDef(
    collection="enrollments",
    domain="education",
    fields=[
        FieldDef(name="enrollment_id", type="string", role=FieldRole.identifier, description="Enrollment ID"),
        FieldDef(name="student_id", type="string", role=FieldRole.identifier, description="Student reference"),
        FieldDef(name="grade", type="double", role=FieldRole.measure, description="Final grade (0-100)"),
        FieldDef(name="enrolled_at", type="date", role=FieldRole.timestamp, description="Enrollment date"),
        FieldDef(name="course", type="string", role=FieldRole.category, description="Course name"),
        FieldDef(name="status", type="string", role=FieldRole.enum, description="Enrollment status",
                 enum_values=["active", "completed", "dropped", "waitlisted"]),
        FieldDef(name="is_auditing", type="bool", role=FieldRole.boolean, description="Auditing (no credit)"),
    ],
)

# ---------------------------------------------------------------------------
# Held-out schemas (NEVER appear in training set)
# ---------------------------------------------------------------------------
MUSEUM_EXHIBITS = SchemaDef(
    collection="museum_exhibits",
    domain="culture",
    fields=[
        FieldDef(name="exhibit_id", type="string", role=FieldRole.identifier, description="Exhibit identifier"),
        FieldDef(name="title", type="string", role=FieldRole.text, description="Exhibit title"),
        FieldDef(name="visitor_count", type="int", role=FieldRole.measure, description="Total visitors"),
        FieldDef(name="opened_on", type="date", role=FieldRole.timestamp, description="Opening date"),
        FieldDef(name="wing", type="string", role=FieldRole.category, description="Museum wing"),
        FieldDef(name="medium", type="string", role=FieldRole.enum, description="Art medium",
                 enum_values=["painting", "sculpture", "photography", "installation", "mixed_media"]),
        FieldDef(name="is_permanent", type="bool", role=FieldRole.boolean, description="Permanent exhibit"),
    ],
)

WEATHER_STATIONS = SchemaDef(
    collection="weather_stations",
    domain="meteorology",
    fields=[
        FieldDef(name="station_id", type="string", role=FieldRole.identifier, description="Station identifier"),
        FieldDef(name="temperature_c", type="double", role=FieldRole.measure, description="Temperature in Celsius"),
        FieldDef(name="recorded_at", type="date", role=FieldRole.timestamp, description="Recording timestamp"),
        FieldDef(name="region", type="string", role=FieldRole.category, description="Geographic region"),
        FieldDef(name="condition", type="string", role=FieldRole.enum, description="Weather condition",
                 enum_values=["clear", "cloudy", "rain", "snow", "storm"]),
        FieldDef(name="humidity_pct", type="double", role=FieldRole.measure, description="Relative humidity %"),
    ],
)

FLEET_VEHICLES = SchemaDef(
    collection="fleet_vehicles",
    domain="transportation",
    fields=[
        FieldDef(name="vehicle_id", type="string", role=FieldRole.identifier, description="Vehicle identifier"),
        FieldDef(name="mileage", type="int", role=FieldRole.measure, description="Odometer reading"),
        FieldDef(name="last_service", type="date", role=FieldRole.timestamp, description="Last service date"),
        FieldDef(name="vehicle_type", type="string", role=FieldRole.enum, description="Vehicle type",
                 enum_values=["sedan", "suv", "truck", "van", "bus"]),
        FieldDef(name="depot", type="string", role=FieldRole.category, description="Home depot"),
        FieldDef(name="is_available", type="bool", role=FieldRole.boolean, description="Currently available"),
        FieldDef(name="fuel_cost", type="double", role=FieldRole.measure, description="Monthly fuel cost"),
    ],
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
TRAIN_SCHEMAS: list[SchemaDef] = [
    ORDERS, PRODUCTS, CUSTOMERS,
    PATIENT_VISITS, LAB_RESULTS,
    SENSOR_READINGS, DEVICE_LOGS,
    EMPLOYEES, PERFORMANCE_REVIEWS,
    TRANSACTIONS, ACCOUNTS,
    SHIPMENTS, WAREHOUSES,
    POSTS, USER_ACTIVITY,
    ENROLLMENTS,
]

HELD_OUT_SCHEMAS: list[SchemaDef] = [
    MUSEUM_EXHIBITS, WEATHER_STATIONS, FLEET_VEHICLES,
]

ALL_SCHEMAS: list[SchemaDef] = TRAIN_SCHEMAS + HELD_OUT_SCHEMAS

HELD_OUT_COLLECTIONS: set[str] = {s.collection for s in HELD_OUT_SCHEMAS}
