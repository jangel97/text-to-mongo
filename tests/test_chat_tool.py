"""Tests for the interactive query tools (no external services required)."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from bson import ObjectId

# Add tools/ to path so we can import core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from core import (
    _convert_extended_json,
    _json_serializer,
    execute_query,
    load_config,
    resolve_collection,
)

# ---------------------------------------------------------------------------
# Dashboard config fixture
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "demos" / "dashboard" / "config.json"


@pytest.fixture(scope="module")
def dashboard_config():
    return load_config(str(_CONFIG_PATH))


# ---------------------------------------------------------------------------
# Dashboard config tests
# ---------------------------------------------------------------------------


class TestDashboardConfig:
    def test_all_collections_present(self, dashboard_config):
        assert set(dashboard_config["schemas"].keys()) == {
            "products", "drops", "artifacts", "git_repositories",
        }

    def test_schemas_have_required_fields(self, dashboard_config):
        for name, schema in dashboard_config["schemas"].items():
            assert "collection" in schema, f"{name} missing 'collection'"
            assert "domain" in schema, f"{name} missing 'domain'"
            assert "fields" in schema, f"{name} missing 'fields'"
            assert len(schema["fields"]) > 0, f"{name} has no fields"

    def test_field_structure(self, dashboard_config):
        for name, schema in dashboard_config["schemas"].items():
            for field in schema["fields"]:
                assert "name" in field, f"{name}: field missing 'name'"
                assert "type" in field, f"{name}.{field.get('name')}: missing 'type'"
                assert "role" in field, f"{name}.{field.get('name')}: missing 'role'"

    def test_short_descriptions(self, dashboard_config):
        """Descriptions must be 2-5 words or the LoRA model hallucinates."""
        for name, schema in dashboard_config["schemas"].items():
            for field in schema["fields"]:
                desc = field.get("description", "")
                if desc:
                    word_count = len(desc.split())
                    assert word_count <= 5, (
                        f"{name}.{field['name']}: description '{desc}' has {word_count} words (max 5)"
                    )

    def test_allowed_ops_not_empty(self, dashboard_config):
        assert len(dashboard_config["allowed_ops"]["stage_operators"]) > 0
        assert len(dashboard_config["allowed_ops"]["expression_operators"]) > 0

    def test_keywords_cover_all_collections(self, dashboard_config):
        assert set(dashboard_config["collection_keywords"].keys()) == set(dashboard_config["schemas"].keys())

    def test_suggestions_not_empty(self, dashboard_config):
        assert len(dashboard_config.get("suggestions", [])) > 0


# ---------------------------------------------------------------------------
# Collection resolver tests (using dashboard keywords)
# ---------------------------------------------------------------------------


class TestCollectionResolver:
    @pytest.fixture(autouse=True)
    def _load_keywords(self, dashboard_config):
        self.keywords = dashboard_config["collection_keywords"]

    def test_artifacts_keywords(self):
        assert resolve_collection("show latest containers", self.keywords) == "artifacts"
        assert resolve_collection("find all disk-images", self.keywords) == "artifacts"
        assert resolve_collection("how many artifacts?", self.keywords) == "artifacts"
        assert resolve_collection("list all wheels", self.keywords) == "artifacts"

    def test_drops_keywords(self):
        assert resolve_collection("show all drops", self.keywords) == "drops"
        assert resolve_collection("latest release", self.keywords) == "drops"
        assert resolve_collection("when was it published?", self.keywords) == "drops"

    def test_products_keywords(self):
        assert resolve_collection("list all products", self.keywords) == "products"
        assert resolve_collection("supported versions for this product", self.keywords) == "products"
        assert resolve_collection("which konflux namespace?", self.keywords) == "products"

    def test_git_repositories_keywords(self):
        assert resolve_collection("list all repositories", self.keywords) == "git_repositories"
        assert resolve_collection("show git repos", self.keywords) == "git_repositories"
        assert resolve_collection("find gitlab projects", self.keywords) == "git_repositories"

    def test_no_match(self):
        assert resolve_collection("hello world", self.keywords) is None
        assert resolve_collection("what is the weather?", self.keywords) is None

    def test_case_insensitive(self):
        assert resolve_collection("Show Latest CONTAINERS", self.keywords) == "artifacts"
        assert resolve_collection("LIST ALL DROPS", self.keywords) == "drops"

    def test_highest_score_wins(self):
        assert resolve_collection("container image build", self.keywords) == "artifacts"

    def test_artifact_in_production(self):
        assert resolve_collection("show latest rhaiis artifact in production", self.keywords) == "artifacts"


# ---------------------------------------------------------------------------
# Generic function tests (no config dependency)
# ---------------------------------------------------------------------------


class TestConvertExtendedJson:
    def test_date_conversion(self):
        obj = {"$date": "2024-01-15T00:00:00Z"}
        result = _convert_extended_json(obj)
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_date_without_z(self):
        obj = {"$date": "2024-06-01T12:00:00+00:00"}
        result = _convert_extended_json(obj)
        assert isinstance(result, datetime)

    def test_nested_date_in_filter(self):
        query = {
            "type": "find",
            "filter": {
                "created_at": {"$gte": {"$date": "2024-01-01T00:00:00Z"}}
            }
        }
        result = _convert_extended_json(query)
        assert isinstance(result["filter"]["created_at"]["$gte"], datetime)

    def test_non_date_passthrough(self):
        obj = {"type": "find", "filter": {"status": "active"}}
        result = _convert_extended_json(obj)
        assert result == obj

    def test_list_passthrough(self):
        obj = [{"$date": "2024-01-01T00:00:00Z"}, "hello"]
        result = _convert_extended_json(obj)
        assert isinstance(result[0], datetime)
        assert result[1] == "hello"

    def test_invalid_date_passthrough(self):
        obj = {"$date": "not-a-date"}
        result = _convert_extended_json(obj)
        assert result == {"$date": "not-a-date"}


class TestJsonSerializer:
    def test_objectid(self):
        oid = ObjectId("507f1f77bcf86cd799439011")
        assert _json_serializer(oid) == "507f1f77bcf86cd799439011"

    def test_datetime(self):
        dt = datetime(2024, 1, 15, 12, 0, 0)
        result = _json_serializer(dt)
        assert "2024-01-15" in result

    def test_fallback_to_str(self):
        assert _json_serializer(42) == "42"
        assert _json_serializer(None) == "None"


class TestExecuteQuery:
    """Tests using a mock MongoDB collection."""

    class MockCursor:
        def __init__(self, docs):
            self._docs = docs
            self._limit = None

        def sort(self, key_or_list):
            return self

        def limit(self, n):
            self._limit = n
            return self

        def __iter__(self):
            docs = self._docs[:self._limit] if self._limit else self._docs
            return iter(docs)

        def __list__(self):
            return list(self.__iter__())

    class MockCollection:
        def __init__(self, docs):
            self._docs = docs
            self.last_pipeline = None

        def find(self, filter_doc=None, projection=None):
            return TestExecuteQuery.MockCursor(self._docs)

        def aggregate(self, pipeline):
            self.last_pipeline = pipeline
            return TestExecuteQuery.MockCursor(self._docs)

    class MockDB:
        def __init__(self, docs):
            self._docs = docs
            self._collection = TestExecuteQuery.MockCollection(self._docs)

        def __getitem__(self, name):
            return self._collection

    def test_find_query(self):
        docs = [{"_id": "1", "name": "test"}]
        db = self.MockDB(docs)
        query = {"type": "find", "filter": {"name": "test"}}
        result = execute_query(db, "artifacts", query)
        assert len(result) == 1
        assert result[0]["name"] == "test"

    def test_aggregate_query(self):
        docs = [{"_id": "product_a", "count": 5}]
        db = self.MockDB(docs)
        query = {"type": "aggregate", "pipeline": [{"$group": {"_id": "$product_key", "count": {"$sum": 1}}}]}
        result = execute_query(db, "artifacts", query)
        assert len(result) == 1

    def test_aggregate_injects_limit(self):
        docs = [{"_id": "a"}]
        db = self.MockDB(docs)
        query = {"type": "aggregate", "pipeline": [{"$group": {"_id": "$type"}}]}
        execute_query(db, "artifacts", query)
        # $limit is injected into the internal copy passed to aggregate
        assert {"$limit": 50} in db._collection.last_pipeline

    def test_aggregate_preserves_existing_limit(self):
        docs = [{"_id": "a"}]
        db = self.MockDB(docs)
        query = {"type": "aggregate", "pipeline": [{"$group": {"_id": "$type"}}, {"$limit": 10}]}
        execute_query(db, "artifacts", query)
        # Should NOT inject another $limit
        limit_stages = [s for s in db._collection.last_pipeline if "$limit" in s]
        assert len(limit_stages) == 1

    def test_unknown_query_type(self):
        db = self.MockDB([])
        query = {"type": "update", "filter": {}}
        result = execute_query(db, "artifacts", query)
        assert result[0]["error"] == "Unknown query type: update"

    def test_find_with_sort_dict(self):
        docs = [{"_id": "1"}, {"_id": "2"}]
        db = self.MockDB(docs)
        query = {"type": "find", "filter": {}, "sort": {"created_at": -1}}
        result = execute_query(db, "artifacts", query)
        assert len(result) == 2

    def test_find_respects_limit_cap(self):
        docs = [{"_id": str(i)} for i in range(100)]
        db = self.MockDB(docs)
        query = {"type": "find", "filter": {}, "limit": 999}
        result = execute_query(db, "artifacts", query)
        assert len(result) <= 50

    def test_date_conversion_in_query(self):
        docs = [{"_id": "1"}]
        db = self.MockDB(docs)
        query = {
            "type": "find",
            "filter": {"created_at": {"$gte": {"$date": "2024-01-01T00:00:00Z"}}}
        }
        # Should not raise — $date gets converted to datetime
        result = execute_query(db, "artifacts", query)
        assert isinstance(result, list)
