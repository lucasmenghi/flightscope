from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    m=json.loads((ROOT/'reports/metrics.json').read_text());f=json.loads((ROOT/'reports/frozen_config.json').read_text());d=json.loads((ROOT/'reports/data_manifest.json').read_text());n=m['selected'];b=m['logistic_baseline'];ci=m['ap_delta_day_bootstrap95']
    table='| Modelo | AP | ROC AUC | Brier ↓ | Precision@20% | Recall@20% | Lift@20% |\n|---|---:|---:|---:|---:|---:|---:|\n'+''.join(f'| {name} | {v["average_precision"]:.3f} | {v["roc_auc"]:.3f} | {v["brier"]:.3f} | {v["precision_at20"]:.1%} | {v["recall_at20"]:.1%} | {v["lift_at20"]:.2f}× |\n' for name,v in [(f['selected'],n),('Logistic baseline',b)])
    content=f'''# FlightScope — Airline Delay Intelligence

![Capa executiva](assets/flightscope-cover.webp)

Classificação do risco de atraso na partida com dados oficiais do BTS. Projeto independente de Lucas Menghi, sem vínculo ou endosso do BTS ou de companhias aéreas.

**[Estudo PT](https://lucasmenghi.github.io/portifolio/projects/flightscope.html)** · **[English study](https://lucasmenghi.github.io/portifolio/projects/flightscope-en.html)** · [Notebook executado](notebooks/01_flight_delay_case.ipynb) · [Model card](docs/model_card.md)

## Problema de negócio

Priorizar voos para acompanhamento operacional quando a capacidade é limitada. Alvo: `DepDel15 = 1`, **partida atrasada em 15 minutos ou mais**, não atraso na chegada. Um ranking de risco pode apoiar triagem, mas não prova redução de atrasos ou retorno financeiro.

## Dados e disponibilidade

[BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/tables.asp?qo_vq=EFD), 12 arquivos oficiais mensais de 2025, {sum(x['source_rows'] for x in d['sources']):,} registros. Amostra de 15.000 menores chaves SHA256 por mês, independente do alvo, total de 180.000. Identificador: data, companhia, número do voo, origem, destino e horário programado. Cancelamentos excluídos; desvios mantidos quando a partida é observada.

Features: companhia, aeroportos, horário programado, mês, dia da semana, distância e duração programada. Horário/mês cíclicos. Sem horários reais, causas de atraso, taxiamento ou cancelamento como entrada. Os horários programados no arquivo retrospectivo não são snapshots auditados da informação conhecida na previsão.

## Protocolo temporal

1. Desenvolvimento: jan–abr→mai–jun e jan–jun→jul–ago. Quatro configurações (duas regressões logísticas, dois LightGBM). Seleção pela AP média.
2. Modelo final: treino jan–ago (120.000 voos), sem refit nos meses seguintes.
3. Setembro (15.000): calibração logística dos logits.
4. Outubro (15.000): limiar no percentil 80 dos scores calibrados, congelado por modelo.
5. Novembro–dezembro (30.000): teste final, sem ajuste após os resultados.

Selecionado: `{f['selected']}`. Seed 42. Modelos, encoders, calibradores e limiares salvos. [Configuração congelada](reports/frozen_config.json) · [Janelas de desenvolvimento](reports/development.csv) · [Manifesto e hashes](reports/data_manifest.json).

## Resultados executados

{table}

Prevalência de atraso no teste: {m['test_delay_rate']:.1%}. Diferença de AP selecionado − referência: {n['average_precision']-b['average_precision']:+.3f}; bootstrap pareado por dia 95%: [{ci[0]:+.3f}, {ci[1]:+.3f}]. 300 reamostragens em blocos de dia; dependência entre dias não modelada.

No **limiar fixo** {n['threshold']:.3f}, a fila de teste foi {n['queue_rate']:.1%}, precision {n['precision']:.1%}, recall {n['recall']:.1%}. Esse limiar não garante capacidade de 20% em outros períodos. As métricas top-20% são um diagnóstico histórico separado: selecionam exatamente os 20% de maiores scores.

[Métricas](reports/metrics.json) · [Estabilidade mensal](reports/monthly_metrics.json) · [Cenários históricos de capacidade](reports/capacity_scenarios.json).

**Decisão:** ganho de ranking modesto; o limiar congelado perdeu cobertura no período posterior. Não recomendar alertas autônomos. Revalidar em outro ano e testar novos dados conhecidos na previsão antes de um piloto. Nenhum ajuste foi feito usando o teste final.

## Reprodução

Python 3.11, ambiente virtual novo recomendado:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare.py
python scripts/train.py
python -m unittest discover -s tests -v
python scripts/build_report.py
python scripts/build_docs.py
streamlit run app.py
```

Download público (~350 MB ZIP), sem chave. Arquivos brutos e processados ficam locais e não são redistribuídos. Os modelos treinados e as opções de entrada estão incluídos: o app roda sem baixar novamente a base. Carregue apenas os artefatos confiáveis deste repo; `joblib` não é um formato seguro para arquivos desconhecidos.

O notebook pode ser executado depois da preparação. Gráficos e relatórios publicados são resultados reais dos scripts. `scripts/build_site.py` integra especificamente os checkouts irmãos do portfólio de Lucas; não é necessário para reproduzir o estudo ou rodar o app.

## Limitações e decisão

Um ano, uma semente e voos domésticos dos EUA. Amostra igualmente ponderada por mês, não uma estimativa ponderada do tráfego nacional. Sem previsão meteorológica, condições em tempo real, congestionamento ou rotação da aeronave. Classificação condicionada a voos operados; não prevê cancelamentos. Calibração pode deteriorar com mudanças sazonais. Repetir em outro ano e validar dados conhecidos no instante da previsão antes de operar. Testar o efeito das ações em piloto, não confundir lift de ranking com ROI.

## Fontes e autoria

- [Base e documentação BTS](https://www.transtats.bts.gov/tables.asp?qo_vq=EFD).
- [Dicionário oficial](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ).
- Código original: MIT. Os dados têm origem federal nos EUA; o projeto não reivindica propriedade dos dados.
- [Prompt da arte](docs/art_prompt.md), ferramenta integrada imagegen, capa conceitual com estética editorial. Não é fotografia documental.
'''
    (ROOT/'README.md').write_text(content,encoding='utf-8')
    card=f'''# Model card — FlightScope

## Modelo e uso

Selecionado `{f['selected']}`, a partir de quatro candidatos e duas janelas temporais pela AP média. Dados BTS 2025; alvo partida ≥15 minutos atrasada. Apoio à triagem humana em voos operados, não decisão autônoma nem status de voo atual. Não usar como previsão de cancelamento ou atraso na chegada.

## Features e treinamento

Entradas brutas: `{f['raw_inputs']}`. Horário e mês cíclicos. Categóricas com encoder treinado somente no treino, categorias desconhecidas tratadas pelo pipeline. Regressões com one-hot e numéricas padronizadas; LightGBM com categóricas ordinais declaradas como categóricas. Seed 42. Hiperparâmetros completos em `reports/frozen_config.json`.

Jan–ago: treino. Setembro: calibração de logits. Outubro: limiar percentil 80. Novembro–dezembro: teste final. Amostra de 15 mil voos por mês, por menores SHA256 do identificador e sem estratificação por alvo. Campos de desfecho excluídos. Campos programados retrospectivos não foram auditados contra snapshots no instante da previsão.

## Avaliação

{table}

Bootstrap em blocos por dia: {ci}, diferença de AP. Correlação intradia parcialmente representada, dependência entre dias não modelada. Uma semente não quantifica variabilidade de treino. Limiar fixo {n['threshold']:.3f}; fila {n['queue_rate']:.1%}; precision {n['precision']:.1%}; recall {n['recall']:.1%}. Ranking top-20% é avaliação separada do limiar congelado.

## Limites e monitoramento

EUA, um ano, meses igualmente amostrados. Não cobre voos cancelados ou sem partida observada. Sem clima previsto ou dados em tempo real. Scores calibrados historicamente não garantem probabilidade individual. Validar em outro ano e monitorar Brier/reliability, precision, recall, prevalência, tamanho da fila e estabilidade por aeroporto/companhia. Só avaliar economia e eficácia com um piloto controlado.

## Fonte e licenças

Registros oficiais BTS; hashes e URLs no manifesto. Código original MIT; bibliotecas mantêm licenças próprias. Capa conceitual gerada com IA. Sem afiliação ou endosso por órgãos públicos ou companhias aéreas.
'''
    (ROOT/'docs/model_card.md').write_text(card,encoding='utf-8')
if __name__=='__main__':main()
