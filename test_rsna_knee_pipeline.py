import tempfile
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import rsna_knee_pipeline as pipeline


class LabelMergeTests(unittest.TestCase):
    def test_precedence_masking_and_soft_qwen(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gold_rows = []
            parser_rows = []
            qwen_rows = []
            for uid in ("gold", "silver", "blocked", "qwen"):
                gold_rows.append(
                    {
                        "StudyInstanceUID": uid,
                        **{target: "" for target in pipeline.TARGETS},
                    }
                )
                parser_rows.append({"StudyInstanceUID": uid})
                qwen_rows.append({"StudyInstanceUID": uid})
            gold_rows[0]["ACL"] = "1"
            parser_rows[0].update({"ACL": "0", "ACL_status": "GOLD"})
            parser_rows[1].update(
                {"ACL": "0", "ACL_status": "SILVER", "ACL_confidence": "0.90"}
            )
            parser_rows[2].update(
                {"ACL": "1", "ACL_status": "CONFLICT", "ACL_confidence": "0.99"}
            )
            parser_rows[3].update({"ACL_status": "UNRESOLVED"})
            qwen_rows[2].update({"ACL_LLM": "0.98", "ACL_SOURCE": "LLM_POS"})
            qwen_rows[3].update({"ACL_LLM": "0.82", "ACL_SOURCE": "LLM_POS"})
            pd.DataFrame(gold_rows).to_csv(root / "train.csv", index=False)
            pd.DataFrame(parser_rows).to_csv(root / "parser.csv", index=False)
            pd.DataFrame(qwen_rows).to_csv(root / "qwen.csv", index=False)

            merged, _ = pipeline.merge_labels(
                root / "train.csv", root / "parser.csv", root / "qwen.csv"
            )
            acl = merged[merged.target == "ACL"].set_index("StudyInstanceUID")
            self.assertEqual(acl.loc["gold", "label_source"], "GOLD")
            self.assertEqual(float(acl.loc["gold", "label_value"]), 1.0)
            self.assertEqual(acl.loc["silver", "label_source"], "SILVER")
            self.assertEqual(acl.loc["blocked", "label_status"], "CONFLICT")
            self.assertTrue(bool(acl.loc["blocked", "is_masked"]))
            self.assertEqual(acl.loc["qwen", "label_source"], "QWEN")
            self.assertAlmostEqual(float(acl.loc["qwen", "label_value"]), 0.82)

    def test_parser_confidence_threshold_and_invalid_value_are_masked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame(
                [{"StudyInstanceUID": "a", **{target: "" for target in pipeline.TARGETS}}]
            ).to_csv(root / "train.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "StudyInstanceUID": "a",
                        "ACL": "1",
                        "ACL_status": "SILVER",
                        "ACL_confidence": "0.84",
                    }
                ]
            ).to_csv(root / "parser.csv", index=False)
            merged, _ = pipeline.merge_labels(root / "train.csv", root / "parser.csv", None)
            row = merged[(merged.StudyInstanceUID == "a") & (merged.target == "ACL")].iloc[0]
            self.assertFalse(bool(row.is_accepted))
            self.assertIsNone(row.label_value)

    def test_parser_research_uncertain_status_is_preserved_and_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame(
                [{"StudyInstanceUID": "a", **{target: "" for target in pipeline.TARGETS}}]
            ).to_csv(root / "train.csv", index=False)
            pd.DataFrame(
                [{"StudyInstanceUID": "a", "ACL": "", "ACL_status": "UNRESOLVED"}]
            ).to_csv(root / "parser.csv", index=False)
            pd.DataFrame(
                [{"StudyInstanceUID": "a", "ACL_NLP": "", "ACL_NLP_status": "UNCERTAIN"}]
            ).to_csv(root / "research.csv", index=False)
            merged, stats = pipeline.merge_labels(
                root / "train.csv",
                root / "parser.csv",
                None,
                parser_research_csv=root / "research.csv",
            )
            row = merged[(merged.StudyInstanceUID == "a") & (merged.target == "ACL")].iloc[0]
            self.assertEqual(row.label_status, "UNCERTAIN")
            self.assertEqual(int(stats.loc[stats.target == "ACL", "excluded_uncertain"].iloc[0]), 1)

    def test_invalid_nonempty_gold_value_is_not_replaced_by_weak_label(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame(
                [
                    {
                        "StudyInstanceUID": "a",
                        **{target: "" for target in pipeline.TARGETS},
                        "ACL": "2",
                    }
                ]
            ).to_csv(root / "train.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "StudyInstanceUID": "a",
                        "ACL": "1",
                        "ACL_status": "SILVER",
                        "ACL_confidence": "0.99",
                    }
                ]
            ).to_csv(root / "parser.csv", index=False)
            merged, _ = pipeline.merge_labels(root / "train.csv", root / "parser.csv", None)
            row = merged[(merged.StudyInstanceUID == "a") & (merged.target == "ACL")].iloc[0]
            self.assertEqual(row.label_status, "INVALID")
            self.assertFalse(bool(row.is_accepted))


class LossTests(unittest.TestCase):
    def test_masked_loss_ignores_nan_masked_targets(self):
        logits = torch.zeros((1, len(pipeline.TARGETS)), requires_grad=True)
        targets = torch.full_like(logits, float("nan"))
        targets[0, 0] = 1.0
        mask = torch.zeros_like(logits)
        mask[0, 0] = 1.0
        weights = torch.ones_like(logits)
        loss = pipeline.masked_bce_loss(logits, targets, mask, weights)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), np.log(2.0), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_masked_loss_rejects_invalid_active_target(self):
        logits = torch.zeros((1, len(pipeline.TARGETS)))
        targets = torch.zeros_like(logits)
        targets[0, 0] = 1.5
        mask = torch.zeros_like(logits)
        mask[0, 0] = 1.0
        with self.assertRaises(ValueError):
            pipeline.masked_bce_loss(logits, targets, mask, torch.ones_like(mask))

    def test_resnet34_synthetic_forward_backward_optimizer_step(self):
        model = pipeline.ResNet34StudyModel(pretrained=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        images = torch.randn(1, 2, 3, 64, 64)
        targets = torch.zeros((1, len(pipeline.TARGETS)))
        mask = torch.zeros_like(targets)
        mask[0, 0] = 1.0

        logits = model(images)
        loss = pipeline.masked_bce_loss(logits, targets, mask, torch.ones_like(mask))
        loss.backward()
        optimizer.step()

        self.assertEqual(tuple(logits.shape), (1, len(pipeline.TARGETS)))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

    def test_auc_is_unavailable_without_both_gold_classes(self):
        self.assertTrue(math.isnan(pipeline._auc(np.array([1, 1]), np.array([0.2, 0.8]))))
        self.assertTrue(math.isnan(pipeline._auc(np.array([0, 0]), np.array([0.2, 0.8]))))


class SplitAndDicomTests(unittest.TestCase):
    def test_group_split_has_no_study_or_patient_leakage(self):
        labels = {}
        ids = [f"study-{i}" for i in range(10)]
        for index, uid in enumerate(ids):
            values = np.zeros(len(pipeline.TARGETS), dtype=np.float32)
            mask = np.zeros_like(values)
            weights = np.zeros_like(values)
            gold_mask = np.zeros_like(values)
            values[0] = float(index % 2)
            mask[0] = weights[0] = gold_mask[0] = 1.0
            labels[uid] = (values, mask, weights, gold_mask)
        patients = {uid: f"patient-{index // 2}" for index, uid in enumerate(ids)}
        train, val, _ = pipeline.grouped_multilabel_split(
            ids, labels, patients, validation_fraction=0.2, seed=9
        )
        self.assertFalse(set(train) & set(val))
        self.assertFalse({patients[uid] for uid in train} & {patients[uid] for uid in val})

    def test_dicom_sort_uses_geometry(self):
        def instance(name, z, number):
            return pipeline.DicomInstance(
                path=name,
                study_uid="study",
                series_uid="series",
                sop_uid=name,
                modality="MR",
                series_description="",
                protocol_name="",
                manufacturer="",
                manufacturer_model="",
                magnetic_field_strength="",
                instance_number=number,
                number_of_frames=1,
                pixel_data_present=True,
                rows=10,
                columns=10,
                pixel_spacing=(1.0, 1.0),
                slice_thickness=1.0,
                spacing_between_slices=1.0,
                image_position=(0.0, 0.0, z),
                image_orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                photometric_interpretation="MONOCHROME2",
                bits_stored=16,
                pixel_representation=0,
                rescale_slope=1.0,
                rescale_intercept=0.0,
                transfer_syntax_uid="",
                patient_group="",
            )

        ordered, method = pipeline.sort_series_instances(
            [instance("higher", 2.0, 1), instance("lower", 1.0, 9)]
        )
        self.assertEqual([record.path for record, _ in ordered], ["lower", "higher"])
        self.assertIn("ImagePositionPatient", method)


class SubmissionTests(unittest.TestCase):
    def test_submission_matches_template_shape_and_order(self):
        template = pd.DataFrame(
            [{"StudyInstanceUID": "b", **{target: 0.0 for target in pipeline.TARGETS}}]
        )
        predictions = pd.DataFrame(
            [{"StudyInstanceUID": "b", **{target: 0.5 for target in pipeline.TARGETS}}]
        )
        result = pipeline.validate_submission(template, predictions)
        self.assertEqual(result.columns.tolist(), template.columns.tolist())
        self.assertEqual(len(result), 1)
        self.assertTrue((result[pipeline.TARGETS] == 0.5).all().all())

    def test_submission_rejects_id_mismatch(self):
        template = pd.DataFrame(
            [{"StudyInstanceUID": "official", **{target: 0.0 for target in pipeline.TARGETS}}]
        )
        predictions = pd.DataFrame(
            [{"StudyInstanceUID": "other", **{target: 0.5 for target in pipeline.TARGETS}}]
        )
        with self.assertRaises(ValueError):
            pipeline.validate_submission(template, predictions)


class KaggleInputDiscoveryTests(unittest.TestCase):
    def test_label_csvs_are_discoverable_from_nested_dataset_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            competition = root / "competition"
            labels = root / "datasets" / "bokkyn" / "labels-final"
            (competition / "train_images").mkdir(parents=True)
            (competition / "test_images").mkdir()
            labels.mkdir(parents=True)
            for name in ("train.csv", "train_series.csv", "test_series.csv", "sample_submission.csv"):
                (competition / name).write_text("", encoding="utf-8")
            for name in ("labels_final.csv", "labels_research.csv"):
                (labels / name).write_text("", encoding="utf-8")

            inputs = pipeline.discover_inputs(root)

            self.assertEqual(inputs["parser_csv"], labels / "labels_final.csv")
            self.assertEqual(inputs["parser_research_csv"], labels / "labels_research.csv")
            self.assertEqual(inputs["train_csv"], competition / "train.csv")
            self.assertEqual(inputs["sample_submission"], competition / "sample_submission.csv")


if __name__ == "__main__":
    unittest.main()
