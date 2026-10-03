from pathlib import Path
import sys,json,joblib,pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from src.core import predict
st.set_page_config(page_title='FlightScope',layout='centered')
st.image(str(ROOT/'assets/flightscope-cover.webp'))
st.title('FlightScope — risco de atraso na partida')
st.write('Modelo histórico para voos domésticos dos EUA. Alvo: partida com atraso de 15 minutos ou mais, condicionado ao voo ser operado.')
@st.cache_resource
def load():return joblib.load(ROOT/'models/selected.joblib')
@st.cache_data
def choices():return json.loads((ROOT/'reports/input_choices.json').read_text())
options=choices()
with st.form('flight'):
    carrier=st.selectbox('Companhia (código BTS)',options['carriers'])
    origin=st.selectbox('Origem',options['airports'],index=options['airports'].index('JFK') if 'JFK' in options['airports'] else 0)
    destination=st.selectbox('Destino',options['airports'],index=options['airports'].index('LAX') if 'LAX' in options['airports'] else 0)
    month=st.selectbox('Mês',range(1,13));weekday=st.selectbox('Dia da semana (1=segunda, 7=domingo)',range(1,8))
    departure=st.time_input('Horário previsto de partida (local)')
    distance=st.number_input('Distância (milhas)',min_value=1,max_value=6000,value=2475)
    elapsed=st.number_input('Duração prevista (minutos)',min_value=1,max_value=900,value=360)
    submitted=st.form_submit_button('Estimar risco')
if submitted:
    if origin==destination:st.error('Selecione aeroportos diferentes.')
    else:
        frame=pd.DataFrame([{'Reporting_Airline':carrier,'Origin':origin,'Dest':destination,'Month':month,'DayOfWeek':weekday,'CRSDepTime':departure.hour*100+departure.minute,'Distance':distance,'CRSElapsedTime':elapsed}])
        bundle=load();score=float(predict(bundle,frame)[0]);st.metric('Risco estimado de atraso ≥15 min',f'{score:.1%}')
        st.write('Acima do limiar histórico de alerta' if score>=bundle['threshold'] else 'Abaixo do limiar histórico de alerta')
        st.caption(f'Limiar congelado: {bundle["threshold"]:.3f}. Estimativa histórica calibrada em setembro de 2025, sem garantia de probabilidade individual.')
st.caption('Sem clima previsto, condições em tempo real ou rotação da aeronave. Não prevê cancelamentos. Sem vínculo com BTS ou companhias aéreas. Não é um serviço de status de voo.')
