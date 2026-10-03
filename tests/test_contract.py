from pathlib import Path
import sys,json,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd,numpy as np
from src.core import features,FEATURES,RAW_INPUTS
class DataContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.df=pd.read_json(ROOT/'data/processed/flights.jsonl',lines=True)
    def test_unique_flights(self):self.assertFalse(self.df.key.duplicated().any())
    def test_temporal_partitions(self):
        groups=[self.df[self.df.split==s] for s in ['train','calibration','policy','test']]
        for a,b in zip(groups,groups[1:]):self.assertLess(a.FlightDate.max(),b.FlightDate.min())
    def test_schedule_only(self):
        self.assertEqual(list(features(self.df.head()).columns),FEATURES)
        self.assertFalse(set(RAW_INPUTS)&{'DepDelay','DepDel15','DepTime','ArrDelay','Cancelled','WeatherDelay','TaxiOut'})
    def test_monthly_sampling(self):
        self.assertTrue((self.df.groupby('Month').size()==15000).all())
    def test_targets_and_inputs(self):
        self.assertEqual(set(self.df.label),{0,1});self.assertTrue(np.isfinite(features(self.df.head())[FEATURES[3:]]).all().all())
class Artifact(unittest.TestCase):
    def test_result_counts(self):
        m=json.loads((ROOT/'reports/metrics.json').read_text())
        for name in ['selected','logistic_baseline']:
            tn,fp=m[name]['confusion_matrix'][0];fn,tp=m[name]['confusion_matrix'][1]
            self.assertEqual(tn+fp+fn+tp,m['test_n']);self.assertAlmostEqual(tp/(tp+fn),m[name]['recall']);self.assertAlmostEqual((tp+fp)/m['test_n'],m[name]['queue_rate'])
    def test_freeze(self):
        m=json.loads((ROOT/'reports/metrics.json').read_text());f=json.loads((ROOT/'reports/frozen_config.json').read_text())
        self.assertEqual(f['selected'],max(m['development_means'],key=m['development_means'].get))
        for name in ['selected','logistic_baseline']:self.assertEqual(f['thresholds'][name],m[name]['threshold'])
    def test_offline_inference(self):
        import joblib
        from src.core import predict
        bundle=joblib.load(ROOT/'models/selected.joblib');p=predict(bundle,pd.read_json(ROOT/'data/processed/flights.jsonl',lines=True).head(3))
        self.assertTrue(((p>=0)&(p<=1)).all())
if __name__=='__main__':unittest.main()
