# FlightScope — Airline Delay Intelligence

![Capa executiva](assets/flightscope-cover.webp)

Classificação do risco de atraso na partida com dados oficiais do BTS. Projeto independente de Lucas Menghi, sem vínculo ou endosso do BTS ou de companhias aéreas.

**[Estudo PT](https://lucasmenghi.github.io/portifolio/projects/flightscope.html)** · **[English study](https://lucasmenghi.github.io/portifolio/projects/flightscope-en.html)** · [Notebook executado](notebooks/01_flight_delay_case.ipynb) · [Model card](docs/model_card.md)

## Problema de negócio

Priorizar voos para acompanhamento operacional quando a capacidade é limitada. Alvo: `DepDel15 = 1`, **partida atrasada em 15 minutos ou mais**, não atraso na chegada. Um ranking de risco pode apoiar triagem, mas não prova redução de atrasos ou retorno financeiro.

## Dados e disponibilidade

[BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/tables.asp?qo_vq=EFD), 12 arquivos oficiais mensais de 2025, 7,001,619 registros. Amostra de 15.000 menores chaves SHA256 por mês, independente do alvo, total de 180.000. Identificador: data, companhia, número do voo, origem, destino e horário programado. Cancelamentos excluídos; desvios mantidos quando a partida é observada.

Features: companhia, aeroportos, horário programado, mês, dia da semana, distância e duração programada. Horário/mês cíclicos. Sem horários reais, causas de atraso, taxiamento ou cancelamento como entrada. Os horários programados no arquivo retrospectivo não são snapshots auditados da informação conhecida na previsão.

## Protocolo temporal

1. Desenvolvimento: jan–abr→mai–jun e jan–jun→jul–ago. Quatro configurações (duas regressões logísticas, dois LightGBM). Seleção pela AP média.
2. Modelo final: treino jan–ago (120.000 voos), sem refit nos meses seguintes.
3. Setembro (15.000): calibração logística dos logits.
4. Outubro (15.000): limiar no percentil 80 dos scores calibrados, congelado por modelo.
5. Novembro–dezembro (30.000): teste final, sem ajuste após os resultados.

Selecionado: `lightgbm_15`. Seed 42. Modelos, encoders, calibradores e limiares salvos. [Configuração congelada](reports/frozen_config.json) · [Janelas de desenvolvimento](reports/development.csv) · [Manifesto e hashes](reports/data_manifest.json).

## Resultados executados

| Modelo | AP | ROC AUC | Brier ↓ | Precision@20% | Recall@20% | Lift@20% |
|---|---:|---:|---:|---:|---:|---:|
| lightgbm_15 | 0.343 | 0.655 | 0.178 | 37.0% | 31.8% | 1.59× |
| Logistic baseline | 0.330 | 0.644 | 0.181 | 35.4% | 30.5% | 1.52× |


Prevalência de atraso no teste: 23.3%. Diferença de AP selecionado − referência: +0.013; bootstrap pareado por dia 95%: [+0.004, +0.023]. 300 reamostragens em blocos de dia; dependência entre dias não modelada.

No **limiar fixo** 0.251, a fila de teste foi 7.1%, precision 41.6%, recall 12.8%. Esse limiar não garante capacidade de 20% em outros períodos. As métricas top-20% são um diagnóstico histórico separado: selecionam exatamente os 20% de maiores scores.

[Métricas](reports/metrics.json) · [Estabilidade mensal](reports/monthly_metrics.json) · [Cenários históricos de capacidade](reports/capacity_scenarios.json).

**Decisão:** ganho de ranking modesto; o limiar congelado perdeu cobertura no período posterior. Não recomendar alertas autônomos. Revalidar em outro ano e testar novos dados conhecidos na previsão antes de um piloto. Nenhum ajuste foi feito usando o teste final.

## Reprodução

Python 3.11, ambiente virtual novo recomendado:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
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
