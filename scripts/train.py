"""Rolling development, separate calibration/policy months, frozen final test."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score,confusion_matrix
from lightgbm import LGBMClassifier
from src.core import CAT,NUM,features,logit,predict,RAW_INPUTS
CANDIDATES={'logistic_c01':{'kind':'logistic','C':.1},'logistic_c1':{'kind':'logistic','C':1.},'lightgbm_15':{'kind':'lightgbm','leaves':15,'trees':250},'lightgbm_31':{'kind':'lightgbm','leaves':31,'trees':400}}
def make_model(config):
    if config['kind']=='logistic':
        prep=ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore',min_frequency=20),CAT),('numeric',StandardScaler(),NUM)])
        model=LogisticRegression(C=config['C'],max_iter=600,random_state=42)
    else:
        prep=ColumnTransformer([('categorical',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CAT),('numeric','passthrough',NUM)])
        model=LGBMClassifier(n_estimators=config['trees'],num_leaves=config['leaves'],learning_rate=.04,min_child_samples=100,reg_lambda=2.,n_jobs=4,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True)
    return Pipeline([('prep',prep),('model',model)])
def fit(config,frame):
    model=make_model(config);kwargs={'model__categorical_feature':[0,1,2]} if config['kind']=='lightgbm' else {}
    model.fit(features(frame),frame.label,**kwargs);return model
def metrics(y,p,threshold):
    y=np.asarray(y);pred=p>=threshold;order=np.argsort(-p,kind='stable');selected=order[:int(np.ceil(.2*len(y)))];precision20=float(y[selected].mean());prevalence=float(y.mean())
    return {'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'queue_rate':float(pred.mean()),'threshold':float(threshold),'precision_at20':precision20,'recall_at20':float(y[selected].sum()/y.sum()),'lift_at20':precision20/prevalence,'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}
def main():
    start=time.time();frame=pd.read_json(ROOT/'data/processed/flights.jsonl',lines=True);rows=[]
    for name,config in CANDIDATES.items():
        for train_end,valid_end in [(4,6),(6,8)]:
            train=frame[frame.Month<=train_end];valid=frame[frame.Month.between(train_end+1,valid_end)]
            model=fit(config,train);p=model.predict_proba(features(valid))[:,1]
            row={'model':name,'train_end_month':train_end,'validation_months':[train_end+1,valid_end],'average_precision':float(average_precision_score(valid.label,p)),'roc_auc':float(roc_auc_score(valid.label,p))};rows.append(row);print(row,flush=True)
    cv=pd.DataFrame(rows);means=cv.groupby('model').average_precision.mean().sort_values(ascending=False);chosen=str(means.index[0])
    train=frame[frame.split=='train'];cal=frame[frame.split=='calibration'];policy=frame[frame.split=='policy'];test=frame[frame.split=='test']
    bundles={};policy_results={}
    for name,candidate in [('selected',chosen),('logistic_baseline','logistic_c1')]:
        model=fit(CANDIDATES[candidate],train);raw=model.predict_proba(features(cal))[:,1]
        calibrator=LogisticRegression(C=1e6,max_iter=1000,random_state=42).fit(logit(raw),cal.label)
        bundle={'model':model,'calibrator':calibrator,'candidate':candidate,'raw_inputs':RAW_INPUTS}
        pp=predict(bundle,policy);bundle['threshold']=float(np.quantile(pp,.80));bundles[name]=bundle;policy_results[name]=metrics(policy.label,pp,bundle['threshold'])
    frozen={'seed':42,'candidates':CANDIDATES,'selected':chosen,'selection':'Mean average precision across two expanding temporal development windows','train_months':[1,8],'calibration_month':9,'policy_month':10,'test_months':[11,12],'threshold_rule':'80th percentile of October calibrated scores, separately per model; fixed thereafter','thresholds':{k:v['threshold'] for k,v in bundles.items()},'raw_inputs':RAW_INPUTS,'target':'DepDel15 = 1, departure delay >=15 minutes'}
    (ROOT/'reports/frozen_config.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');cv.to_csv(ROOT/'reports/development.csv',index=False)
    # The test is first scored after model, calibration and thresholds are frozen.
    predictions={k:predict(v,test) for k,v in bundles.items()};results={k:metrics(test.label,p,bundles[k]['threshold']) for k,p in predictions.items()}
    rng=np.random.default_rng(42);y=test.label.to_numpy();days=test.FlightDate.astype(str).to_numpy();unique=np.unique(days);byday={d:np.flatnonzero(days==d) for d in unique};delta=[]
    for _ in range(300):
        ix=np.concatenate([byday[d] for d in rng.choice(unique,len(unique),replace=True)]);delta.append(average_precision_score(y[ix],predictions['selected'][ix])-average_precision_score(y[ix],predictions['logistic_baseline'][ix]))
    results.update({'test_n':len(test),'test_delay_rate':float(test.label.mean()),'policy_metrics':policy_results,'development_means':means.to_dict(),'ap_delta_day_bootstrap95':np.quantile(delta,[.025,.975]).tolist(),'bootstrap_note':'300 paired calendar-day block resamples; serial dependence between days is not modeled','runtime_seconds':round(time.time()-start,1)})
    report=test[['key','FlightDate','Month','Reporting_Airline','label']].copy()
    for name,p in predictions.items():report[name+'_score']=p
    report.to_json(ROOT/'reports/test_predictions.jsonl',orient='records',lines=True,date_format='iso')
    for name,bundle in bundles.items():joblib.dump(bundle,ROOT/f'models/{name}.joblib')
    segments=[]
    for month,g in report.groupby('Month'):
        row={'month':int(month),'n':len(g),'delay_rate':float(g.label.mean())};row.update(metrics(g.label,g.selected_score.to_numpy(),bundles['selected']['threshold']));segments.append(row)
    (ROOT/'reports/monthly_metrics.json').write_text(json.dumps(segments,indent=2),encoding='utf-8')
    (ROOT/'reports/metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
