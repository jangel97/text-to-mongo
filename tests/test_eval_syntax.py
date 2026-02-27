import json

import pytest

from text_to_mongo.eval.syntax import eval_syntax


class TestValidJson:
    def test_valid_aggregate(self):
        query = json.dumps({
            "type": "aggregate",
            "pipeline": [{"$match": {"status": "active"}}],
        })
        result = eval_syntax(query)
        assert result.passed
        assert result.valid_json
        assert result.has_type
        assert result.type_value == "aggregate"
        assert result.has_body
        assert result.pipeline_well_formed

    def test_valid_find(self):
        query = json.dumps({
            "type": "find",
            "filter": {"status": "active"},
        })
        result = eval_syntax(query)
        assert result.passed
        assert result.valid_json
        assert result.type_value == "find"

    def test_multi_stage_pipeline(self):
        query = json.dumps({
            "type": "aggregate",
            "pipeline": [
                {"$match": {"status": "active"}},
                {"$group": {"_id": "$dept", "total": {"$sum": "$amount"}}},
                {"$sort": {"total": -1}},
            ],
        })
        result = eval_syntax(query)
        assert result.passed
        assert result.pipeline_well_formed


class TestInvalidJson:
    def test_not_json(self):
        result = eval_syntax("this is not json")
        assert not result.passed
        assert not result.valid_json
        assert "Invalid JSON" in result.errors[0]

    def test_empty_string(self):
        result = eval_syntax("")
        assert not result.passed

    def test_json_array(self):
        result = eval_syntax("[1, 2, 3]")
        assert not result.passed
        assert result.valid_json
        assert "Top-level" in result.errors[0]


class TestMissingFields:
    def test_no_type(self):
        query = json.dumps({"pipeline": [{"$match": {}}]})
        result = eval_syntax(query)
        assert not result.passed
        assert "Missing 'type'" in result.errors[0]

    def test_wrong_type(self):
        query = json.dumps({"type": "update", "filter": {}})
        result = eval_syntax(query)
        assert not result.passed
        assert "Invalid type" in result.errors[0]

    def test_aggregate_no_pipeline(self):
        query = json.dumps({"type": "aggregate"})
        result = eval_syntax(query)
        assert not result.passed
        assert "missing 'pipeline'" in result.errors[0]

    def test_find_no_filter(self):
        query = json.dumps({"type": "find"})
        result = eval_syntax(query)
        assert not result.passed
        assert "missing 'filter'" in result.errors[0]


class TestMalformedPipeline:
    def test_pipeline_not_list(self):
        query = json.dumps({"type": "aggregate", "pipeline": "oops"})
        result = eval_syntax(query)
        assert not result.passed
        assert "'pipeline' must be a list" in result.errors[0]

    def test_stage_not_dict(self):
        query = json.dumps({"type": "aggregate", "pipeline": ["not a dict"]})
        result = eval_syntax(query)
        assert not result.passed

    def test_stage_no_dollar_key(self):
        query = json.dumps({"type": "aggregate", "pipeline": [{"match": {}}]})
        result = eval_syntax(query)
        assert not result.passed
        assert "exactly one $-prefixed key" in result.errors[0]

    def test_stage_multiple_dollar_keys(self):
        query = json.dumps({
            "type": "aggregate",
            "pipeline": [{"$match": {}, "$sort": {}}],
        })
        result = eval_syntax(query)
        assert not result.passed

    def test_empty_pipeline(self):
        query = json.dumps({"type": "aggregate", "pipeline": []})
        result = eval_syntax(query)
        assert result.passed  # empty pipeline is structurally valid
