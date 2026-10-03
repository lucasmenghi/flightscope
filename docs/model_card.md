# Model card — FlightScope

## Modelo e uso

Selecionado `lightgbm_15`, a partir de quatro candidatos e duas janelas temporais pela AP média. Dados BTS 2025; alvo partida ≥15 minutos atrasada. Apoio à triagem humana em voos operados, não decisão autônoma nem status de voo atual. Não usar como previsão de cancelamento ou atraso na chegada.

## Features e treinamento

Entradas brutas: `['Reporting_Airline', 'Origin', 'Dest', 'CRSDepTime', 'Month', 'DayOfWeek', 'Distance', 'CRSElapsedTime']`. Horário e mês cíclicos. Categóricas com encoder treinado somente no treino, categorias desconhecidas tratadas pelo pipeline. Regressões com one-hot e numéricas padronizadas; LightGBM com categóricas ordinais declaradas como categóricas. Seed 42. Hiperparâmetros completos em `reports/frozen_config.json`.

Jan–ago: treino. Setembro: calibração de logits. Outubro: limiar percentil 80. Novembro–dezembro: teste final. Amostra de 15 mil voos por mês, por menores SHA256 do identificador e sem estratificação por alvo. Campos de desfecho excluídos. Campos programados retrospectivos não foram auditados contra snapshots no instante da previsão.

## Avaliação

| Modelo | AP | ROC AUC | Brier ↓ | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| lightgbm_15 | 0.343 | 0.655 | 0.178 | 37.0% | 31.8% | 1.59× |
| Logistic baseline | 0.330 | 0.644 | 0.181 | 35.4% | 30.5% | 1.52× |


Bootstrap em blocos por dia: [0.0037872793527500464, 0.023036313424898276], diferença de AP. Correlação intradia parcialmente representada, dependência entre dias não modelada. Uma semente não quantifica variabilidade de treino. Limiar fixo 0.251; fila 7.1%; precision 41.6%; recall 12.8%. Ranking top-20% é avaliação separada do limiar congelado.

## Limites e monitoramento

EUA, um ano, meses igualmente amostrados. Não cobre voos cancelados ou sem partida observada. Sem clima previsto ou dados em tempo real. Scores calibrados historicamente não garantem probabilidade individual. Validar em outro ano e monitorar Brier/reliability, precision, recall, prevalência, tamanho da fila e estabilidade por aeroporto/companhia. Só avaliar economia e eficácia com um piloto controlado.

## Fonte e licenças

Registros oficiais BTS; hashes e URLs no manifesto. Código original MIT; bibliotecas mantêm licenças próprias. Capa conceitual gerada com IA. Sem afiliação ou endosso por órgãos públicos ou companhias aéreas.
