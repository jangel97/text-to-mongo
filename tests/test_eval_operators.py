import pytest

from text_to_mongo.eval.operators import eval_operators, extract_operators


class TestExtractOperators:
    def test_simple_match(self):
        obj = {"$match": {"status": "active"}}
        assert "$match" in extract_operators(obj)

    def test_nested_operators(self):
        obj = {
            "$group": {
                "_id": "$dept",
                "total": {"$sum": "$amount"},
                "avg_price": {"$avg": "$price"},
            }
        }
        ops = extract_operators(obj)
        assert ops == {"$group", "$sum", "$avg"}

    def test_list_of_stages(self):
        pipeline = [
            {"$match": {"status": "active"}},
            {"$group": {"_id": "$dept", "total": {"$sum": 1}}},
            {"$sort": {"total": -1}},
        ]
        ops = extract_operators(pipeline)
        assert "$match" in ops
        assert "$group" in ops
        assert "$sort" in ops
        assert "$sum" in ops

    def test_deeply_nested(self):
        obj = {"$match": {"$or": [{"x": {"$gt": 5}}, {"y": {"$lt": 10}}]}}
        ops = extract_operators(obj)
        assert ops == {"$match", "$or", "$gt", "$lt"}

    def test_no_operators(self):
        obj = {"name": "test", "value": 42}
        assert extract_operators(obj) == set()


class TestEvalOperators:
    def test_all_allowed(self):
        query = {
            "type": "aggregate",
            "pipeline": [
                {"$match": {"status": "active"}},
                {"$group": {"_id": "$dept", "total": {"$sum": "$amount"}}},
            ],
        }
        allowed = ["$match", "$group", "$sum"]
        result = eval_operators(query, allowed)
        assert result.passed
        assert len(result.violations) == 0
        assert len(result.unsafe_operators) == 0

    def test_violation(self):
        query = {
            "type": "aggregate",
            "pipeline": [
                {"$match": {"status": "active"}},
                {"$unwind": "$items"},
            ],
        }
        allowed = ["$match"]
        result = eval_operators(query, allowed)
        assert not result.passed
        assert "$unwind" in result.violations

    def test_unsafe_operator(self):
        query = {
            "type": "aggregate",
            "pipeline": [
                {"$match": {"$where": "this.x > 10"}},
            ],
        }
        allowed = ["$match", "$where"]
        result = eval_operators(query, allowed)
        assert not result.passed
        assert "$where" in result.unsafe_operators

    def test_merge_blocked(self):
        query = {
            "type": "aggregate",
            "pipeline": [
                {"$merge": {"into": "output_collection"}},
            ],
        }
        allowed = ["$merge"]
        result = eval_operators(query, allowed)
        assert not result.passed
        assert "$merge" in result.unsafe_operators

    def test_mixed_violations_and_unsafe(self):
        query = {
            "type": "aggregate",
            "pipeline": [
                {"$match": {}},
                {"$out": "bad"},
                {"$lookup": {"from": "other"}},
            ],
        }
        allowed = ["$match"]
        result = eval_operators(query, allowed)
        assert not result.passed
        assert "$out" in result.unsafe_operators
        assert "$lookup" in result.violations
