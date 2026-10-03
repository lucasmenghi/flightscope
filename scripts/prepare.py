"""Official monthly BTS sources, deterministic label-independent sampling."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests,zipfile,hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE='https://www.transtats.bts.gov/PREZIP/'
COLS=['FlightDate','Reporting_Airline','Flight_Number_Reporting_Airline','Origin','Dest','CRSDepTime','CRSElapsedTime','Distance','DayOfWeek','Month','DepDel15','Cancelled']
def download(month):
    filename=f'On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{month}.zip'
    path=ROOT/'data/raw'/filename;url=BASE+filename
    if not path.exists():
        temp=path.with_suffix('.part')
        with requests.get(url,stream=True,timeout=(30,180)) as r:
            r.raise_for_status()
            with temp.open('wb') as f:
                for chunk in r.iter_content(1024*1024):f.write(chunk)
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:raise ValueError('Corrupt source archive')
        temp.replace(path)
    return month,path,url
def prepare(source):
    month,path,url=source;pieces=[];source_n=0;eligible_n=0
    with zipfile.ZipFile(path) as z:
        csv=next(n for n in z.namelist() if n.lower().endswith('.csv'))
        with z.open(csv) as f:
            for frame in pd.read_csv(f,usecols=COLS,chunksize=60000):
                source_n+=len(frame)
                frame=frame[(frame.Cancelled==0)&frame.DepDel15.isin([0,1])].dropna(subset=COLS)
                valid_time=(frame.CRSDepTime.between(0,2400))&((frame.CRSDepTime.astype(int)%100)<60)
                frame=frame[valid_time&(frame.CRSElapsedTime>0)&(frame.Distance>0)].copy();eligible_n+=len(frame)
                keys=frame[['FlightDate','Reporting_Airline','Flight_Number_Reporting_Airline','Origin','Dest','CRSDepTime']].astype(str).agg('|'.join,axis=1)
                frame['key']=keys.map(lambda k:hashlib.sha256(k.encode()).hexdigest())
                pieces.append(frame.sort_values('key').head(15000))
    sample=pd.concat(pieces).sort_values('key').drop_duplicates('key').head(15000)
    sample['label']=sample.DepDel15.astype(int)
    sample=sample.drop(columns=['DepDel15','Cancelled'])
    info={'month':month,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_rows':source_n,'eligible_rows':eligible_n,'sample_rows':len(sample),'delayed_rate':float(sample.label.mean())}
    print(info,flush=True);return sample,info
def main():
    for d in ['data/raw','data/processed','reports','reports/figures','models','docs','assets','notebooks','tests','tmp']:(ROOT/d).mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:sources=list(pool.map(download,range(1,13)))
    results=[prepare(s) for s in sources];frame=pd.concat([r[0] for r in results]).sort_values(['FlightDate','key']).reset_index(drop=True)
    frame['split']='train';frame.loc[frame.Month==9,'split']='calibration';frame.loc[frame.Month==10,'split']='policy';frame.loc[frame.Month>=11,'split']='test'
    frame.to_json(ROOT/'data/processed/flights.jsonl',orient='records',lines=True)
    manifest={'source':'BTS Reporting Carrier On-Time Performance','year':2025,'sample_rule':'15,000 smallest SHA256 flight keys per month; no label stratification','target':'Departure delay >=15 minutes (DepDel15)','scope':'US reported domestic operated flights with observed departure label and valid scheduled fields; cancellations excluded; diversions retained if departure observed','sources':[r[1] for r in results],'splits':{s:{'n':len(g),'start':g.FlightDate.min(),'end':g.FlightDate.max(),'delay_rate':float(g.label.mean())} for s,g in frame.groupby('split')}}
    (ROOT/'reports/data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Prepared',len(frame),'flights',flush=True)
if __name__=='__main__':main()
