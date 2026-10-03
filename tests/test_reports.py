from pathlib import Path
import json,unittest
ROOT=Path(__file__).resolve().parents[1]
class Reports(unittest.TestCase):
    def test_counts(self):
        m=json.loads((ROOT/'reports/metrics.json').read_text())
        for name in ['selected','logistic_baseline']:
            tn,fp=m[name]['confusion_matrix'][0];fn,tp=m[name]['confusion_matrix'][1]
            self.assertEqual(tn+fp+fn+tp,m['test_n']);self.assertAlmostEqual(tp/(tp+fn),m[name]['recall'])
    def test_frozen_thresholds(self):
        m=json.loads((ROOT/'reports/metrics.json').read_text());f=json.loads((ROOT/'reports/frozen_config.json').read_text())
        for name in ['selected','logistic_baseline']:self.assertEqual(f['thresholds'][name],m[name]['threshold'])
    def test_capacity(self):
        m=json.loads((ROOT/'reports/metrics.json').read_text());scenarios=json.loads((ROOT/'reports/capacity_scenarios.json').read_text())
        scenario=next(x for x in scenarios if x['capacity']==20)
        self.assertEqual(scenario['selected']['queue'],round(.2*m['test_n']))
        self.assertAlmostEqual(scenario['selected']['precision'],m['selected']['precision_at20'])
if __name__=='__main__':unittest.main()
