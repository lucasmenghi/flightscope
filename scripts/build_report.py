from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve
import nbformat as nb
def main():
    m=json.loads((ROOT/'reports/metrics.json').read_text());pred=pd.read_json(ROOT/'reports/test_predictions.jsonl',lines=True);dev=pd.read_csv(ROOT/'reports/development.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    out=ROOT/'reports/figures';out.mkdir(exist_ok=True)
    for kind in ['precision_recall','calibration']:
        fig,ax=plt.subplots(figsize=(9,5))
        for name,label,color in [('selected','Selected classifier','#244575'),('logistic_baseline','Logistic baseline','#9aa9bd')]:
            p=pred[name+'_score']
            if kind=='precision_recall':
                precision,recall,_=precision_recall_curve(pred.label,p);ax.plot(recall,precision,label=label,color=color,lw=2)
            else:
                observed,forecast=calibration_curve(pred.label,p,n_bins=10,strategy='quantile');ax.plot(forecast,observed,marker='o',label=label,color=color,lw=2)
        if kind=='precision_recall':ax.axhline(m['test_delay_rate'],ls='--',color='#c1cbd9',label='Delay prevalence');ax.set(xlabel='Recall',ylabel='Precision',title='Nov–Dec 2025 final test • precision–recall')
        else:ax.plot([0,1],[0,1],ls='--',color='#c1cbd9');ax.set(xlabel='Mean calibrated forecast',ylabel='Observed delay rate',title='Final-test reliability • quantile bins')
        ax.legend();fig.tight_layout();fig.savefig(out/(kind+'.png'),dpi=160);plt.close(fig)
    means=dev.groupby('model').average_precision.mean().sort_values();fig,ax=plt.subplots(figsize=(9,4));ax.barh(means.index,means.values,color='#244575');ax.set(xlabel='Mean validation average precision',title='Four candidates • two expanding temporal windows');fig.tight_layout();fig.savefig(out/'development.png',dpi=160);plt.close(fig)
    months=json.loads((ROOT/'reports/monthly_metrics.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(['Nov','Dec'],[x['average_precision'] for x in months],color='#244575');axes[0].set(title='Final-test average precision',ylim=(0,1))
    axes[1].bar(['Nov','Dec'],[x['queue_rate'] for x in months],color='#7f9abd');axes[1].axhline(.2,color='#244575',ls='--');axes[1].set(title='Queue at frozen October threshold',ylim=(0,1));fig.tight_layout();fig.savefig(out/'monthly_stability.png',dpi=160);plt.close(fig)
    scenarios=[]
    for percent in range(5,51,5):
        row={'capacity':percent}
        for name in ['selected','logistic_baseline']:
            selected=pred.sort_values(name+'_score',ascending=False,kind='stable').head(int(np.ceil(len(pred)*percent/100)))
            row[name]={'queue':len(selected),'identified':int(selected.label.sum()),'precision':float(selected.label.mean()),'recall':float(selected.label.sum()/pred.label.sum()),'lift':float(selected.label.mean()/pred.label.mean())}
        scenarios.append(row)
    (ROOT/'reports/capacity_scenarios.json').write_text(json.dumps(scenarios,indent=2),encoding='utf-8')
    frame=pd.read_json(ROOT/'data/processed/flights.jsonl',lines=True);train=frame[frame.split=='train'];choices={'carriers':sorted(train.Reporting_Airline.unique().tolist()),'airports':sorted(set(train.Origin)|set(train.Dest))}
    (ROOT/'reports/input_choices.json').write_text(json.dumps(choices,indent=2),encoding='utf-8')
    book=nb.v4.new_notebook();book.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}}
    book.cells=[nb.v4.new_markdown_cell('# FlightScope — classificação de atrasos aéreos\n\nCase independente de Lucas Menghi com registros oficiais do BTS. CRISP-DM: estimar risco de partida ≥15 minutos atrasada para apoiar triagem operacional. Não é um serviço atual de status de voo.'),nb.v4.new_code_cell("from pathlib import Path\nimport json,pandas as pd\nROOT=Path.cwd() if (Path.cwd()/'reports').exists() else Path.cwd().parent\nmanifest=json.loads((ROOT/'reports/data_manifest.json').read_text())\npd.DataFrame(manifest['splits']).T"),nb.v4.new_markdown_cell('## Compreensão e preparação dos dados\n12 arquivos mensais oficiais de 2025, SHA256 registrado. Voos domésticos reportados e operados, com DepDel15 observado e campos programados válidos. Amostra de 15.000 menores hashes por mês, sem estratificar pelo alvo. Cancelamentos excluídos; desvios mantidos quando a partida é observada. Os pesos de meses são iguais na amostra, não proporcionais ao tráfego nacional.'),nb.v4.new_code_cell("# Reproduzir download e preparação (~350 MB comprimidos)\n# %run ../scripts/prepare.py\ndf=pd.read_json(ROOT/'data/processed/flights.jsonl',lines=True)\ndf.groupby('Month').agg(n=('label','size'),delay_rate=('label','mean'))"),nb.v4.new_markdown_cell('## Modelagem sem variáveis de desfecho\nSomente companhia, origem/destino, horário programado, mês, dia da semana, distância e duração programada. Horário e mês têm codificação cíclica. Não usar partida/chegada reais, causas de atraso, taxiamento ou status de cancelamento como entrada. Quatro candidatos (duas regressões logísticas e dois LightGBM), duas janelas expansivas: jan–abr→mai–jun e jan–jun→jul–ago. Seleção por AP média.'),nb.v4.new_code_cell("development=pd.read_csv(ROOT/'reports/development.csv')\ndevelopment.pivot(index='model',columns='train_end_month',values='average_precision')"),nb.v4.new_markdown_cell('## Calibração, limiar e congelamento\nModelo final treinado jan–ago. Setembro: calibração logística do logit dos scores. Outubro: limiar no percentil 80 dos scores calibrados. Novembro–dezembro: teste final, após congelar a configuração. A fila não terá necessariamente 20% em períodos futuros. Ranking top-20% é uma avaliação histórica separada, com capacidade fixa.'),nb.v4.new_code_cell("config=json.loads((ROOT/'reports/frozen_config.json').read_text())\nconfig"),nb.v4.new_code_cell("metrics=json.loads((ROOT/'reports/metrics.json').read_text())\npd.DataFrame({name:{k:v for k,v in metrics[name].items() if k!='confusion_matrix'} for name in ['selected','logistic_baseline']}).T"),nb.v4.new_code_cell("from IPython.display import Image,display\nfor name in ['development','precision_recall','calibration','monthly_stability']:\n    display(Image(filename=str(ROOT/'reports/figures'/f'{name}.png')))"),nb.v4.new_markdown_cell('## Limites e aplicação\nDados históricos dos EUA, amostra com 15 mil voos por mês. Sem clima previsto, congestionamento em tempo real ou rotação da aeronave; não prevê cancelamentos. Campos programados no arquivo retrospectivo não são snapshots auditados do que era conhecido no momento da previsão. Bootstrap em blocos por dia considera correlação intradia, mas não entre dias. Não medir ROI ou redução de atrasos sem piloto. Reavaliar em outro ano/estação e validar a disponibilidade dos campos antes de operar.'),nb.v4.new_code_cell("scenarios=json.loads((ROOT/'reports/capacity_scenarios.json').read_text())\npd.DataFrame([{'capacity':s['capacity'],**s['selected']} for s in scenarios])"),nb.v4.new_markdown_cell('## Fontes\n- https://www.transtats.bts.gov/tables.asp?qo_vq=EFD\n- https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ\n\nRelatórios e figuras foram calculados pelos scripts executados. Cenários de capacidade são diagnósticos históricos e não escolhas adicionais do modelo.')]
    nb.write(book,ROOT/'notebooks/01_flight_delay_case.ipynb')
if __name__=='__main__':main()
