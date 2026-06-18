"""Tests for the ``cli`` package — publish_utils, train_utils, redact, and CLI arg parsing.
These tests were authored by DeepSeek V4 Flash. I have not reviewed them.
its just safety for now
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli.publish_utils import (
    ENTITY_LABELS,
    get_best_checkpoint,
    get_first_checkpoint_args,
    load_training_args,
    make_entity_table,
)


# ===========================================================================
# cli/publish_utils.py
# ===========================================================================


class TestMakeEntityTable:
    def test_all_entities_present(self):
        log = {f"eval_{label}_f1": 0.95 for label in ENTITY_LABELS}
        log.update({f"eval_{label}_support": 100 for label in ENTITY_LABELS})
        result = make_entity_table(log)
        assert result.startswith("| Entity | F1 | Support |")
        assert "EMAIL" in result
        assert "0.9500" in result
        for label in ENTITY_LABELS:
            assert label in result

    def test_missing_entities_skipped(self):
        log = {"eval_EMAIL_f1": 0.9, "eval_EMAIL_support": 50}
        result = make_entity_table(log)
        assert "EMAIL" in result
        assert "BOD" not in result

    def test_worst_n_filter(self):
        log = {f"eval_{label}_f1": i * 0.1 for i, label in enumerate(ENTITY_LABELS)}
        log.update({f"eval_{label}_support": 100 for label in ENTITY_LABELS})
        result = make_entity_table(log, worst=3)
        # Should only contain 3 rows
        row_count = result.strip().count("\n") - 1  # subtract header
        assert row_count == 3

    def test_empty_log(self):
        result = make_entity_table({})
        assert result == "| Entity | F1 | Support |\n|--------|------|---------|\n"

    def test_none_values_skipped(self):
        log = {"eval_EMAIL_f1": None, "eval_EMAIL_support": 50}
        result = make_entity_table(log)
        assert "EMAIL" not in result


class TestGetBestCheckpoint:
    def test_returns_best_checkpoint(self, tmp_path: Path):
        ckpt_dir = tmp_path / "checkpoint-100"
        ckpt_dir.mkdir()
        state = {
            "best_model_checkpoint": str(ckpt_dir),
            "best_global_step": 100,
            "log_history": [
                {"step": 50, "eval_f1": 0.9},
                {"step": 100, "eval_f1": 0.95},
            ],
        }
        (ckpt_dir / "trainer_state.json").write_text(json.dumps(state))

        best_ckpt, best_state, best_log = get_best_checkpoint(tmp_path)
        assert best_ckpt == ckpt_dir
        assert best_state == state
        assert best_log["eval_f1"] == 0.95

    def test_no_checkpoints_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No checkpoint directories"):
            get_best_checkpoint(tmp_path)

    def test_multiple_checkpoints_picks_latest(self, tmp_path: Path):
        (tmp_path / "checkpoint-50").mkdir()
        ckpt_200 = tmp_path / "checkpoint-200"
        ckpt_200.mkdir()
        state = {
            "best_model_checkpoint": str(ckpt_200),
            "best_global_step": 200,
            "log_history": [
                {"step": 200, "eval_f1": 0.96},
                {"step": 50, "eval_f1": 0.90},
            ],
        }
        (ckpt_200 / "trainer_state.json").write_text(json.dumps(state))

        best_ckpt, _, best_log = get_best_checkpoint(tmp_path)
        assert best_ckpt == ckpt_200
        assert best_log["eval_f1"] == 0.96


class TestLoadTrainingArgs:
    def test_loads_training_args(self, tmp_path: Path):
        mock_args = MagicMock()
        mock_args.learning_rate = 1e-3
        mock_args.bf16 = True

        with patch("cli.publish_utils.torch.load", return_value=mock_args) as mock_load:
            result = load_training_args(tmp_path)
            mock_load.assert_called_once_with(
                tmp_path / "training_args.bin", weights_only=False
            )
            assert result.learning_rate == 1e-3

    def test_file_not_found(self, tmp_path: Path):
        with patch("cli.publish_utils.torch.load", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_training_args(tmp_path)


class TestGetFirstCheckpointArgs:
    def test_returns_first_checkpoint(self, tmp_path: Path):
        (tmp_path / "checkpoint-200").mkdir()
        first_dir = tmp_path / "checkpoint-50"
        first_dir.mkdir()
        (tmp_path / "checkpoint-100").mkdir()

        mock_args = MagicMock()
        mock_args.learning_rate = 1e-3
        mock_args.bf16 = True

        with patch("cli.publish_utils.torch.load", return_value=mock_args) as mock_load:
            result = get_first_checkpoint_args(tmp_path)
            mock_load.assert_called_once_with(
                first_dir / "training_args.bin", weights_only=False
            )
            assert result.learning_rate == 1e-3

    def test_no_checkpoints_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No checkpoint directories"):
            get_first_checkpoint_args(tmp_path)


# ===========================================================================
# cli/train_utils.py
# ===========================================================================


class TestWeightedTokenClassificationTrainer:
    def test_compute_loss_with_class_weights(self):
        with patch("cli.train_utils.Trainer.__init__", return_value=None):
            from cli.train_utils import WeightedTokenClassificationTrainer

            weights = MagicMock()
            trainer = WeightedTokenClassificationTrainer(class_weights=weights)

            mock_model = MagicMock()
            mock_logits = MagicMock()
            mock_model.return_value = type("Output", (), {"logits": mock_logits})()

            labels = MagicMock()
            inputs = {"labels": labels}

            with patch("cli.train_utils.nn.CrossEntropyLoss") as mock_loss_cls:
                mock_loss = MagicMock()
                mock_loss_cls.return_value = mock_loss

                _ = trainer.compute_loss(mock_model, inputs)

                mock_loss_cls.assert_called_once_with(
                    weight=weights.to.return_value, ignore_index=-100
                )
                mock_loss.assert_called_once()

    def test_compute_loss_without_labels_returns_model_loss(self):
        with patch("cli.train_utils.Trainer.__init__", return_value=None):
            from cli.train_utils import WeightedTokenClassificationTrainer

            trainer = WeightedTokenClassificationTrainer(class_weights=None)

            mock_model = MagicMock()
            mock_model.training = False
            mock_model.return_value = type("Output", (), {"loss": MagicMock()})()

            loss = trainer.compute_loss(mock_model, {"no_labels": True})
            assert loss is not None

    def test_compute_loss_no_labels_training_raises(self):
        with patch("cli.train_utils.Trainer.__init__", return_value=None):
            from cli.train_utils import WeightedTokenClassificationTrainer

            trainer = WeightedTokenClassificationTrainer(class_weights=None)

            mock_model = MagicMock()
            # Simulate the model being called and outputting something without loss
            mock_model.return_value = type("Output", (), {"loss": None})()

            # Set model to training mode
            mock_model.training = True

            with pytest.raises(ValueError, match="Labels are required during training"):
                trainer.compute_loss(mock_model, {})


class TestDetailedProgressCallback:
    def test_on_log_training_step(self, capsys):
        with patch("cli.train_utils.TrainerCallback.__init__", return_value=None):
            from cli.train_utils import DetailedProgressCallback

            cb = DetailedProgressCallback()

            args = MagicMock()
            state = MagicMock()
            state.global_step = 10
            state.max_steps = 100
            state.is_local_process_zero = True
            logs = {"loss": 0.5, "learning_rate": 1e-4}

            cb.on_log(args, state, MagicMock(), logs=logs)
            captured = capsys.readouterr()
            assert "Step 10/100" in captured.out
            assert "loss: 0.5000" in captured.out
            assert "lr: 1.00e-04" in captured.out

    def test_on_log_eval_step(self, capsys):
        with patch("cli.train_utils.TrainerCallback.__init__", return_value=None):
            from cli.train_utils import DetailedProgressCallback

            cb = DetailedProgressCallback()

            args = MagicMock()
            state = MagicMock()
            state.global_step = 5
            state.is_local_process_zero = True
            logs = {"eval_loss": 0.3, "eval_f1": 0.95}

            cb.on_log(args, state, MagicMock(), logs=logs)
            captured = capsys.readouterr()
            assert "Eval step 5" in captured.out
            assert "eval_loss: 0.3000" in captured.out
            assert "f1: 0.9500" in captured.out

    def test_on_log_skipped_non_zero_process(self, capsys):
        with patch("cli.train_utils.TrainerCallback.__init__", return_value=None):
            from cli.train_utils import DetailedProgressCallback

            cb = DetailedProgressCallback()

            args = MagicMock()
            state = MagicMock()
            state.is_local_process_zero = False
            logs = {"loss": 0.5, "learning_rate": 1e-4}

            cb.on_log(args, state, MagicMock(), logs=logs)
            captured = capsys.readouterr()
            assert captured.out == ""

    def test_on_log_no_logs(self, capsys):
        with patch("cli.train_utils.TrainerCallback.__init__", return_value=None):
            from cli.train_utils import DetailedProgressCallback

            cb = DetailedProgressCallback()

            args = MagicMock()
            state = MagicMock()
            state.is_local_process_zero = True

            cb.on_log(args, state, MagicMock(), logs=None)
            captured = capsys.readouterr()
            assert captured.out == ""


class TestMakeComputeMetricsFn:
    def test_returns_metrics_dict(self):
        with (
            patch("cli.train_utils.evaluate.load") as mock_eval_load,
        ):
            from cli.train_utils import make_compute_metrics_fn

            mock_seqeval = MagicMock()
            mock_seqeval.compute.return_value = {
                "overall_precision": 0.9,
                "overall_recall": 0.85,
                "overall_f1": 0.87,
                "overall_accuracy": 0.95,
                "EMAIL": {"f1": 0.92, "number": 100},
            }
            mock_eval_load.return_value = mock_seqeval

            id2label = {0: "O", 1: "B-EMAIL", 2: "I-EMAIL"}
            fn = make_compute_metrics_fn(id2label)

            predictions = MagicMock()
            predictions.predictions = [[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]]
            predictions.label_ids = [[1, 0]]

            result = fn(predictions)
            assert result["precision"] == 0.9
            assert result["recall"] == 0.85
            assert result["f1"] == 0.87
            assert result["accuracy"] == 0.95
            assert result["EMAIL_f1"] == 0.92
            assert result["EMAIL_support"] == 100


class TestTokenizeAndAlignLabels:
    def test_alignment_basic(self):
        from cli.train_utils import tokenize_and_align_labels

        # Build a mock object that supports both dict access (offset_mapping)
        # and attribute access (word_ids method)
        class MockEncoding:
            def __init__(self):
                self._data = {
                    "input_ids": [[101, 205, 220, 102]],
                    "offset_mapping": [
                        [(0, 0), (0, 3), (4, 8), (0, 0)]
                    ],
                }
                self._word_ids = [None, 0, 1, None]

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

            def word_ids(self, batch_index=0):
                return self._word_ids

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MockEncoding()

        label2id = {"O": 0, "B-EMAIL": 1, "I-EMAIL": 2}

        batch = {
            "source_text": ["test@example.com"],
            "privacy_mask": [
                [
                    {"start": 0, "end": 16, "label": "EMAIL"}
                ]
            ],
        }

        result = tokenize_and_align_labels(batch, mock_tokenizer, label2id)
        assert "labels" in result._data
        assert "ner_tags" in result._data


# ===========================================================================
# cli/redact.py
# ===========================================================================


class TestRedactParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["redact.py", "--text", "hello"]):
            from cli.redact import _parse_args

            args = _parse_args()
            assert args.text == "hello"
            assert args.file is None
            assert args.model_variant == "small"
            assert args.threshold == 0.3
            assert args.output_path is None
            assert args.stride == 0.5
            assert args.max_length == 512

    def test_all_args(self):
        with patch("sys.argv", [
            "redact.py",
            "--file", "/tmp/input.txt",
            "--model_variant", "base",
            "--threshold", "0.5",
            "--output_path", "/tmp/out.json",
            "--stride", "0.25",
            "--max_length", "384",
        ]):
            from cli.redact import _parse_args

            args = _parse_args()
            assert args.file == "/tmp/input.txt"
            assert args.model_variant == "base"
            assert args.threshold == 0.5
            assert args.output_path == "/tmp/out.json"
            assert args.stride == 0.25
            assert args.max_length == 384

    def test_model_variant_choices(self):
        with patch("sys.argv", ["redact.py", "--text", "hi", "--model_variant", "invalid"]):
            from cli.redact import _parse_args

            with pytest.raises(SystemExit):
                _parse_args()


class TestRedactValidateArgs:
    def test_inline_text(self, tmp_path: Path):
        from cli.redact import _validate_args

        args = MagicMock()
        args.text = "hello world"
        args.file = None
        assert _validate_args(args) == "hello world"

    def test_file_input(self, tmp_path: Path):
        from cli.redact import _validate_args

        f = tmp_path / "input.txt"
        f.write_text("file content")
        args = MagicMock()
        args.text = None
        args.file = str(f)
        assert _validate_args(args) == "file content"

    def test_missing_text_and_file(self):
        from cli.redact import _validate_args

        args = MagicMock()
        args.text = None
        args.file = None
        with pytest.raises(SystemExit):
            _validate_args(args)

    def test_mutually_exclusive(self):
        from cli.redact import _validate_args

        args = MagicMock()
        args.text = "inline"
        args.file = "/tmp/file.txt"
        with pytest.raises(SystemExit):
            _validate_args(args)

    def test_empty_text(self):
        from cli.redact import _validate_args

        args = MagicMock()
        args.text = ""
        args.file = None
        with pytest.raises(SystemExit):
            _validate_args(args)


class TestRedactMain:
    def test_main_with_text(self):
        from cli.redact import main

        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"original": "hi"}'

        mock_redactor = MagicMock()
        mock_redactor.predict.return_value = mock_response

        with (
            patch("cli.redact.PIIRedactor", return_value=mock_redactor),
            patch("sys.argv", ["redact.py", "--text", "hello"]),
        ):
            main()
            mock_redactor.predict.assert_called_once_with("hello", threshold=0.3)

    def test_main_with_output_path(self, tmp_path: Path):
        from cli.redact import main

        out = tmp_path / "out.json"
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"original": "hi"}'

        mock_redactor = MagicMock()
        mock_redactor.predict.return_value = mock_response

        with (
            patch("cli.redact.PIIRedactor", return_value=mock_redactor),
            patch("sys.argv", ["redact.py", "--text", "hello", "--output_path", str(out)]),
        ):
            main()
            assert out.read_text() == '{"original": "hi"}'


# ===========================================================================
# cli/prepare_ds.py — parse_args
# ===========================================================================


class TestPrepareDsParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["prepare_ds.py"]):
            from cli.prepare_ds import parse_args

            args = parse_args()
            assert args.dataset_name == "ai4privacy/pii-masking-300k"
            assert args.output_dir is None
            assert args.rare_threshold == 50

    def test_custom_values(self):
        with patch("sys.argv", [
            "prepare_ds.py",
            "--dataset_name", "custom/dataset",
            "--output_dir", "/tmp/data",
            "--rare_threshold", "100",
        ]):
            from cli.prepare_ds import parse_args

            args = parse_args()
            assert args.dataset_name == "custom/dataset"
            assert args.output_dir == "/tmp/data"
            assert args.rare_threshold == 100


# ===========================================================================
# cli/train.py — parse_args
# ===========================================================================


class TestTrainParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["train.py"]):
            from cli.train import parse_args

            args = parse_args()
            assert args.model_variant == "small"
            assert args.dataset_path is None
            assert args.label_info_path is None
            assert args.output_dir is None
            assert args.stage1_epochs == 2
            assert args.stage1_lr == 1e-3
            assert args.stage2_epochs == 10
            assert args.stage2_lr == 2e-5
            assert args.batch_size == 16
            assert args.wandb_project == "pii-redaction"
            assert args.no_wandb is False

    def test_custom_values(self):
        with patch("sys.argv", [
            "train.py",
            "--model_variant", "base",
            "--dataset_path", "/tmp/ds",
            "--stage1_epochs", "3",
            "--stage2_epochs", "8",
            "--batch_size", "8",
            "--no_wandb",
        ]):
            from cli.train import parse_args

            args = parse_args()
            assert args.model_variant == "base"
            assert args.dataset_path == "/tmp/ds"
            assert args.stage1_epochs == 3
            assert args.stage2_epochs == 8
            assert args.batch_size == 8
            assert args.no_wandb is True

    def test_invalid_model_variant(self):
        with patch("sys.argv", ["train.py", "--model_variant", "invalid"]):
            from cli.train import parse_args

            with pytest.raises(SystemExit):
                parse_args()


# ===========================================================================
# cli/publish.py — parse_args
# ===========================================================================


class TestPublishParseArgs:
    def test_defaults(self):
        with patch("sys.argv", ["publish.py"]):
            from cli.publish import parse_args

            args = parse_args()
            assert args.models_dir is None
            assert args.model_cards_dir is None
            assert args.push is False
            assert args.benchmark_only is False

    def test_flags(self):
        with patch("sys.argv", [
            "publish.py",
            "--models_dir", "/tmp/models",
            "--model_cards_dir", "/tmp/cards",
            "--push",
            "--benchmark_only",
        ]):
            from cli.publish import parse_args

            args = parse_args()
            assert args.models_dir == "/tmp/models"
            assert args.model_cards_dir == "/tmp/cards"
            assert args.push is True
            assert args.benchmark_only is True
