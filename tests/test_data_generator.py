import json
import random
from pathlib import Path

import pytest

from text_to_mongo.data.augment import (
    augment_field_names,
    generate_negatives,
    run_all_augmentations,
)
from text_to_mongo.data.generator import generate_base_examples
from text_to_mongo.data.schemas import (
    HELD_OUT_COLLECTIONS,
    HELD_OUT_SCHEMAS,
    TRAIN_SCHEMAS,
)
from text_to_mongo.schema import AllowedOps, FieldDef, FieldRole, SchemaDef, TrainingExample


class TestSchemaTemplateMatching:
    def test_base_examples_non_empty(self):
        examples = generate_base_examples(seed=42)
        assert len(examples) > 100, f"Expected 100+ base examples, got {len(examples)}"

    def test_every_train_schema_has_examples(self):
        examples = generate_base_examples(seed=42)
        collections_seen = {ex.schema_def.collection for ex in examples}
        for schema in TRAIN_SCHEMAS:
            assert schema.collection in collections_seen, (
                f"No examples generated for {schema.collection}"
            )

    def test_held_out_schemas_have_examples(self):
        examples = generate_base_examples(seed=42)
        for schema in HELD_OUT_SCHEMAS:
            matching = [e for e in examples if e.schema_def.collection == schema.collection]
            assert len(matching) > 0, f"No examples for held-out schema {schema.collection}"

    def test_example_structure(self):
        examples = generate_base_examples(seed=42)
        for ex in examples[:20]:
            assert ex.intent, "Intent must not be empty"
            assert "type" in ex.output, "Output must have 'type' field"
            assert ex.output["type"] in ("aggregate", "find"), (
                f"Bad type: {ex.output['type']}"
            )


class TestAugmentation:
    def test_field_name_shuffling(self):
        examples = generate_base_examples(seed=42)
        rng = random.Random(42)
        augmented = augment_field_names(examples, rng, ratio=1.0)
        # Not all examples will have renameable fields, but some should
        assert len(augmented) > 0, "Field name shuffling produced no examples"

        # Verify that at least one augmented example has a different field name
        original_fields = {f.name for ex in examples for f in ex.schema_def.fields}
        aug_fields = {f.name for ex in augmented for f in ex.schema_def.fields}
        new_fields = aug_fields - original_fields
        assert len(new_fields) > 0, "No new field names introduced"

    def test_negative_examples(self):
        examples = generate_base_examples(seed=42)
        rng = random.Random(42)
        negatives = generate_negatives(examples, rng, ratio=1.0)
        assert len(negatives) > 0
        for neg in negatives:
            assert neg.is_negative
            assert "error" in neg.output

    def test_run_all_augmentations(self):
        examples = generate_base_examples(seed=42)
        augmented = run_all_augmentations(examples, seed=42)
        assert len(augmented) > 0, "Augmentation produced no additional examples"


class TestDatasetExport:
    def test_export_creates_files(self, tmp_path: Path):
        from text_to_mongo.data.generator import generate_dataset

        counts = generate_dataset(seed=42, output_dir=tmp_path)

        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.jsonl").exists()
        assert (tmp_path / "held_out.jsonl").exists()

        assert counts["train"] > 0
        assert counts["eval"] > 0
        assert counts["held_out"] > 0

    def test_no_held_out_in_train(self, tmp_path: Path):
        from text_to_mongo.data.generator import generate_dataset

        generate_dataset(seed=42, output_dir=tmp_path)

        with open(tmp_path / "train.jsonl") as f:
            for line in f:
                record = json.loads(line)
                assert record["schema"]["collection"] not in HELD_OUT_COLLECTIONS, (
                    f"Held-out schema {record['schema']['collection']} found in train split"
                )

        with open(tmp_path / "eval.jsonl") as f:
            for line in f:
                record = json.loads(line)
                assert record["schema"]["collection"] not in HELD_OUT_COLLECTIONS

    def test_held_out_only_contains_held_out(self, tmp_path: Path):
        from text_to_mongo.data.generator import generate_dataset

        generate_dataset(seed=42, output_dir=tmp_path)

        with open(tmp_path / "held_out.jsonl") as f:
            for line in f:
                record = json.loads(line)
                assert record["schema"]["collection"] in HELD_OUT_COLLECTIONS

    def test_total_example_count(self, tmp_path: Path):
        from text_to_mongo.data.generator import generate_dataset

        counts = generate_dataset(seed=42, output_dir=tmp_path)
        total = sum(counts.values())
        assert total >= 200, f"Expected 200+ total examples, got {total}"

    def test_jsonl_records_valid(self, tmp_path: Path):
        from text_to_mongo.data.generator import generate_dataset

        generate_dataset(seed=42, output_dir=tmp_path)

        for split in ["train", "eval", "held_out"]:
            with open(tmp_path / f"{split}.jsonl") as f:
                for i, line in enumerate(f):
                    record = json.loads(line)
                    assert "schema" in record, f"{split} line {i}: missing schema"
                    assert "intent" in record, f"{split} line {i}: missing intent"
                    assert "output" in record, f"{split} line {i}: missing output"
                    assert "allowed_ops" in record, f"{split} line {i}: missing allowed_ops"
