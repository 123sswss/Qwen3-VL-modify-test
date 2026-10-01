"""CPU-only artifact binding tests; never imports torch or model code."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from diagnostics.run_five_task_test_suite import audit_v1, bind_pathvqa, paired_rsvqa


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root, seed, suffix, score):
    name = f'pathvqa_v1_norm_fixed_5ep_seed{seed}'
    run = root/'pathvqa/outputs/visual_selection_prefix'/(name+'_'+suffix)
    dump(run/'train_report.json', {'experiment': name, 'dataset': 'PathVQA', 'model_seed': seed,
         'data_seed': 42, 'epochs': 5, 'saved_epochs': [3,4,5], 'visual_prompt_mode': 'split18',
         'total_trainable_parameters': 1864963, 'trainer_model_accepts_loss_kwargs': False,
         'accelerator_gradient_accumulation_steps': 1, 'visual_av10_learning_rate': 1e-4,
         'optimizer': {'per_device_batch_size': 2, 'gradient_accumulation_steps': 16,
                       'warmup_ratio': .03, 'scheduler': 'linear', 'max_grad_norm': 1,
                       'group_learning_rates': {'p20': .3, 'visual_s8': 3e-5, 'visual_av10': 1e-4}}})
    dump(run/'checkpoints/epoch_5/visual_selection_prefix_config.json', {
         'method': 'visual_selection_prefix_p20_v1', 'init_seed': seed, 'visual_tokens': [8,10],
         'prefix_tokens': 20, 'layers': [5,11,17], 'trainable_parameters': {'test': 1864963}})
    with zipfile.ZipFile(run/'checkpoints/epoch_5/visual_selection_prefix.pt', 'w') as archive:
        archive.writestr('fixture/data.pkl', b'fixture-not-real-weights')
    dump(run/'eval_validation/epoch_5/pathvqa_summary.json', {
         'split': 'validation', 'count': 6259, 'overall_accuracy': score})
    return run


class BindingTests(unittest.TestCase):
    def test_original_five_epoch_report_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp),44,'20260926_2',59.3386)
            report = json.loads((run/'train_report.json').read_text())
            report['method'] = 'visual_selection_prefix_p20_v1'
            del report['visual_prompt_mode']
            del report['visual_av10_learning_rate']
            rates = report['optimizer'].pop('group_learning_rates')
            dump(run/'train_report.json', report)
            (run/'train.log').write_text('[V1_OPTIMIZER] rates='+json.dumps(rates)+
                ' warmup_ratio=0.03 scheduler=linear\n', encoding='utf-8')
            result = audit_v1(run,44)
            self.assertEqual(result['compatibility_evidence']['visual_prompt_mode'],'split18')
            self.assertIn('[V1_OPTIMIZER]',result['compatibility_evidence']['rates_source'])
            rates['visual_av10'] = 3e-5
            (run/'train.log').write_text('[V1_OPTIMIZER] rates='+json.dumps(rates)+
                ' warmup_ratio=0.03 scheduler=linear\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                audit_v1(run,44)
            (run/'train.log').unlink()
            with self.assertRaises(ValueError):
                audit_v1(run,44)

    def test_original_five_epoch_report_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp),44,'20260926_2',59.3386)
            report = json.loads((run/'train_report.json').read_text())
            report['method'] = 'visual_selection_prefix_p20_v1'
            del report['visual_prompt_mode']
            del report['visual_av10_learning_rate']
            rates = report['optimizer'].pop('group_learning_rates')
            dump(run/'train_report.json', report)
            (run/'train.log').write_text('[V1_OPTIMIZER] rates='+json.dumps(rates)+
                ' warmup_ratio=0.03 scheduler=linear\n', encoding='utf-8')
            result = audit_v1(run,44)
            self.assertEqual(result['compatibility_evidence']['visual_prompt_mode'],'split18')
            self.assertIn('[V1_OPTIMIZER]',result['compatibility_evidence']['rates_source'])
            rates['visual_av10'] = 3e-5
            (run/'train.log').write_text('[V1_OPTIMIZER] rates='+json.dumps(rates)+
                ' warmup_ratio=0.03 scheduler=linear\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                audit_v1(run,44)
            (run/'train.log').unlink()
            with self.assertRaises(ValueError):
                audit_v1(run,44)

    def test_paired_cluster_statistics_identical_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            types = ['rural_urban','presence','count','comp']
            rows = [{'question_id': i, 'image_id': i % 100, 'question_type': types[(i//100)%4],
                     'question': str(i), 'ground_truth_answer': 'yes',
                     'normalized_ground_truth_answer': 'yes', 'correct': i%2}
                    for i in range(10004)]
            for p in (root/'a',root/'b'):
                dump(p/'rsvqa_comparisons.json',rows)
                dump(p/'rsvqa_summary.json', {'average_accuracy': 50})
            result = paired_rsvqa(root/'a',root/'b')
            self.assertEqual(result['AA_ci95'],[0,0])
            self.assertEqual(result['AA_delta'],0)
            self.assertEqual(result['groups']['overall']['image_clusters'],100)
            for group in result['groups'].values():
                self.assertEqual(group['delta'],0)
                self.assertEqual(group['ci95'],[0,0])
                self.assertEqual(group['variant_only_correct'],0)

    def test_exact_unique_identity_not_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root,44,'20260926_2',59.3386)
            fixture(root,45,'20260926',58.5077)
            fixture(root,46,'20260926',58.7314)
            wrong = fixture(root,45,'20990101',58.5077)
            report = json.loads((wrong/'train_report.json').read_text())
            report['epochs'] = 3
            dump(wrong/'train_report.json', report)
            bindings = bind_pathvqa(root)
            self.assertEqual([b['seed'] for b in bindings], [44,45,46])
            self.assertTrue(bindings[1]['run'].endswith('20260926'))
            self.assertEqual(len(bindings[1]['rejected_candidates']),0)
            fixture(root,45,'20260926_1',58.5077)
            self.assertTrue(bind_pathvqa(root)[1]['run'].endswith('20260926'))

    def test_reject_wrong_normalization_and_visual_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp),44,'20260926_2',59.3386)
            report = json.loads((run/'train_report.json').read_text())
            report['trainer_model_accepts_loss_kwargs'] = True
            dump(run/'train_report.json',report)
            with self.assertRaises(ValueError):
                audit_v1(run,44)

    def test_missing_checkpoint_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp),44,'20260926_2',59.3386)
            (run/'checkpoints/epoch_5/visual_selection_prefix.pt').unlink()
            with self.assertRaises(ValueError):
                audit_v1(run,44)

    def test_truncated_checkpoint_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = fixture(Path(tmp),44,'20260926_2',59.3386)
            (run/'checkpoints/epoch_5/visual_selection_prefix.pt').write_bytes(b'truncated')
            with self.assertRaises(zipfile.BadZipFile):
                audit_v1(run,44)


if __name__ == '__main__':
    unittest.main()
