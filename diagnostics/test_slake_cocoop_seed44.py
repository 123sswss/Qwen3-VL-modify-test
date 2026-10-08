"""CPU-only checks: no torch, Transformers or model imports."""
import json
from pathlib import Path
import tempfile
import unittest

from diagnostics.run_slake_cocoop_seed44 import historical_audit, paired, PROTOCOL, evaluate_slake_predictions


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SlakeCoCoOpProtocolTest(unittest.TestCase):
    def test_historical_unknown_is_not_current_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = {key: PROTOCOL[key] for key in (
                "prompt_length", "bottleneck_dim", "trainable_parameters", "seed", "data_seed",
                "prompt_learning_rate", "meta_net_learning_rate")}
            report.update(experiment="pathvqa_cocoop_style_p20_h160_seed44",
                          dataset="PathVQA", train_metrics={"epoch": 3})
            write(root / "train_report.json", report)
            write(root / "checkpoints/epoch_3/cocoop_prompt_config.json", {
                "init_seed": 44, "hidden_size": 2560, "prompt_length": 20,
                "bottleneck_dim": 160, "question_access": False,
                "visual_source": "post_merger_llm_visual_token_mean",
                "prompt_placement": "before_full_chat"})
            (root / "train.log").write_text(
                "[PATHVQA_COCOOP_STYLE_CONFIG] epochs=3 seed=44 data_seed=42\n", encoding="utf-8")
            audit = historical_audit(root)
            self.assertEqual(audit["runtime_versions"], "unknown")
            self.assertEqual(audit["historical_model_accepts_loss_kwargs"], "unknown")
            self.assertEqual(audit["configuration_evidence"]["batch_size"]["value"], "unknown")
            self.assertEqual(audit["configuration_evidence"]["epochs"]["value"], 3)
            with (root / "train.log").open("a", encoding="utf-8") as handle:
                handle.write("[PATHVQA_COCOOP_STYLE_CONFIG] batch_size=4\n")
            with self.assertRaisesRegex(ValueError, "override"):
                historical_audit(root)

    def test_clustered_pairing_scores_groups_and_rejects_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            refs = [{"qid": i, "question": "q", "answer": "yes", "img_name": f"xmlab{i//2}/source.jpg",
                     "answer_type": "OPEN" if i%2 else "CLOSED",
                     "base_type": "vqa" if i%2 else "kvqa",
                     "q_lang": "en" if i%2 else "zh"} for i in range(8)]
            def save(directory, answers):
                preds = [{"question_id": i, "answer": answer} for i, answer in enumerate(answers)]
                summary, comparisons = evaluate_slake_predictions(refs, preds)
                summary.update(language="all", expected_split="test", partial_evaluation=False,
                    base_types=[], max_new_tokens=32, temperature=0, answer_mode="raw",
                    instruction="language-aware-short-answer")
                write(directory / "slake_predictions.json", preds)
                write(directory / "slake_summary.json", summary)
                write(directory / "slake_comparisons.json", comparisons)
            save(root / "a", ["no"] * 8)
            save(root / "b", ["yes"] * 8)
            result = paired(root / "a", root / "b", refs, iterations=100)
            self.assertEqual(result["groups"]["overall"]["ci95"], [100, 100])
            self.assertEqual(result["groups"]["overall"]["image_clusters"], 4)
            self.assertEqual(result["groups"]["OPEN"]["variant_only_correct"], 4)
            self.assertTrue(result["groups"]["en"]["exploratory"])
            write(root / "b/slake_predictions.json", [{"question_id": 0, "answer": "yes"}])
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                paired(root / "a", root / "b", refs, iterations=100)


if __name__ == "__main__":
    unittest.main()
