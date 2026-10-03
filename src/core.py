"""Only schedule information enters the models."""
import numpy as np
import pandas as pd
CAT=['Reporting_Airline','Origin','Dest']
NUM=['departure_sin','departure_cos','month_sin','month_cos','DayOfWeek','Distance','CRSElapsedTime']
FEATURES=CAT+NUM
RAW_INPUTS=CAT+['CRSDepTime','Month','DayOfWeek','Distance','CRSElapsedTime']
def features(frame):
    out=frame[CAT+['DayOfWeek','Distance','CRSElapsedTime']].copy()
    for c in CAT:out[c]=out[c].astype(str)
    hhmm=frame.CRSDepTime.astype(int)
    minutes=((hhmm//100)*60+hhmm%100)%1440
    out['departure_sin']=np.sin(2*np.pi*minutes/1440)
    out['departure_cos']=np.cos(2*np.pi*minutes/1440)
    out['month_sin']=np.sin(2*np.pi*(frame.Month-1)/12)
    out['month_cos']=np.cos(2*np.pi*(frame.Month-1)/12)
    return out[FEATURES]
def logit(p):
    p=np.clip(p,1e-5,1-1e-5);return np.log(p/(1-p)).reshape(-1,1)
def predict(bundle,frame):
    raw=bundle['model'].predict_proba(features(frame))[:,1]
    return bundle['calibrator'].predict_proba(logit(raw))[:,1]
