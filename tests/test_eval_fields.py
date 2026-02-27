import pytest

from text_to_mongo.eval.fields import eval_fields, extract_field_refs
from text_to_mongo.schema import FieldDef, FieldRole, SchemaDef


def _make_schema(field_names: list[str]) -> SchemaDef:
    return SchemaDef(
        collection="test_collection",
        domain="test",
        fields=[
            FieldDef(name=n, type="string", role=FieldRole.category) for n in field_names
        ],
    )


class TestExtractFieldRefs:
    def test_dollar_ref(self):
        refs = extract_field_refs("$price")
        assert refs == {"price"}

    def test_dotted_path(self):
        refs = extract_field_refs("$addr.city")
        assert refs == {"addr"}

    def test_match_keys(self):
        obj = {"status": "active", "region": "US"}
        refs = extract_field_refs(obj)
        assert refs == {"status", "region"}

    def test_nested_group(self):
        obj = {"$group": {"_id": "$dept", "total": {"$sum": "$amount"}}}
        refs = extract_field_refs(obj)
        assert "dept" in refs
        assert "amount" in refs

    def test_pipeline(self):
        pipeline = [
            {"$match": {"status": "active"}},
            {"$group": {"_id": "$category", "avg": {"$avg": "$price"}}},
        ]
        refs = extract_field_refs(pipeline)
        assert "status" in refs
        assert "category" in refs
        assert "price" in refs

    def test_system_vars_excluded(self):
        # $$ROOT, $$NOW etc. should not be counted as field refs
        refs = extract_field_refs("$$ROOT")
        assert len(refs) == 0

    def test_no_refs(self):
        refs = extract_field_refs(42)
        assert refs == set()

    def test_in_operator(self):
        obj = {"status": {"$in": ["active", "pending"]}}
        refs = extract_field_refs(obj)
        assert "status" in refs


class TestEvalFields:
    def test_all_valid(self):
        schema = _make_schema(["status", "region", "amount"])
        query = {
            "type": "find",
            "filter": {"status": "active"},
        }
        result = eval_fields(query, schema)
        assert result.passed
        assert len(result.hallucinated_fields) == 0

    def test_hallucinated_field(self):
        schema = _make_schema(["status", "region"])
        query = {
            "type": "find",
            "filter": {"nonexistent_field": "value"},
        }
        result = eval_fields(query, schema)
        assert not result.passed
        assert "nonexistent_field" in result.hallucinated_fields

    def test_id_always_valid(self):
        schema = _make_schema(["status"])
        query = {
            "type": "aggregate",
            "pipeline": [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}},
            ],
        }
        result = eval_fields(query, schema)
        assert result.passed
        # _id should not appear in hallucinated fields

    def test_coverage(self):
        schema = _make_schema(["a", "b", "c", "d"])
        query = {
            "type": "find",
            "filter": {"a": 1, "b": 2},
        }
        result = eval_fields(query, schema)
        assert result.coverage == pytest.approx(0.5)

    def test_dotted_path_valid(self):
        schema = _make_schema(["address"])
        query = {
            "type": "find",
            "filter": {"address.city": "NYC"},
        }
        result = eval_fields(query, schema)
        assert result.passed
        assert "address" in result.referenced_fields

    def test_empty_query(self):
        schema = _make_schema(["status"])
        query = {
            "type": "find",
            "filter": {},
        }
        result = eval_fields(query, schema)
        assert result.passed
        assert result.coverage == 0.0

    def test_projection_fields_valid(self):
        schema = _make_schema(["name", "status", "region"])
        query = {
            "type": "find",
            "filter": {},
            "projection": {"name": 1, "status": 1},
        }
        result = eval_fields(query, schema)
        assert result.passed
        assert "name" in result.referenced_fields
        assert "status" in result.referenced_fields

    def test_projection_hallucinated_field(self):
        schema = _make_schema(["name", "status"])
        query = {
            "type": "find",
            "filter": {},
            "projection": {"name": 1, "fake_field": 1},
        }
        result = eval_fields(query, schema)
        assert not result.passed
        assert "fake_field" in result.hallucinated_fields

    def test_projection_with_filter(self):
        schema = _make_schema(["name", "status", "region"])
        query = {
            "type": "find",
            "filter": {"status": "active"},
            "projection": {"name": 1, "region": 1},
        }
        result = eval_fields(query, schema)
        assert result.passed
        assert {"name", "status", "region"} == result.referenced_fields
