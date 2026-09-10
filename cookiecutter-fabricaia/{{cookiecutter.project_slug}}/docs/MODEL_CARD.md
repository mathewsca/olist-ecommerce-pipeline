# Model Card — Modelo de Predição de Risco

Este documento formaliza as características, métricas, limitações e diretrizes de uso do modelo de Machine Learning, seguindo as melhores práticas de governança, auditabilidade e Engenharia de Software para IA (*SE4AI*).

---

## 1. Detalhes do Modelo

* **Nome do Modelo:** {{cookiecutter.project_name}} - Classificador de Risco de Execução Contratual
* **Versão:** 1.0.0
* **Data de Lançamento:** 2026-09
* **Tipo de Modelo:** Classificação Supervisionada Binária (Random Forest / Regressão Logística)
* **Frameworks e Bibliotecas:** Scikit-Learn, MLflow, Pandas, NumPy
* **Repositório de Código:** Template FabricaIA
* **Rastreabilidade de Artefatos:** Registrado e versionado via MLflow Tracking (`sqlite:///mlflow.db` ou servidor remoto) sob o experimento configurado.

---

## 2. Finalidade Pretendida (Intended Use)

### Casos de Uso Apropriados
* **Suporte à Decisão Técnica:** Triagem antecipada e ordenação de prioridade de contratos públicos que apresentam maior probabilidade de paralisação ou atraso severo.
* **Planejamento de Fiscalização:** Alocação inteligente de equipes técnicas de vistoria presencial para frentes de obra com anomalias de execução físico-financeira.
* **Monitoramento Contínuo:** Execução em lote (*batch scoring*) e consultas sob demanda via API REST.

### Usos Fora de Escopo e Não Recomendados
* **Decisões Automatizadas sem Intervenção Humana (*No Black-Box Automated Penalties*):** O modelo **não** deve ser utilizado para aplicar sanções contratuais, rescisões ou retenções de pagamento de forma autônoma sem laudo emitido por especialista técnico.
* **Generalização Indevida:** Aplicação direta em domínios contratuais com dinâmica substancialmente distinta (ex: concessões de longo prazo ou compras de bens comuns) sem calibração prévia.

---

## 3. Atributos de Entrada e Pré-processamento

| Atributo | Tipo | Descrição | Tratamento |
| :--- | :--- | :--- | :--- |
| `tipo_obra` | Categórico | Categoria da intervenção pública | *Ordinal / Label Encoding* |
| `valor_previsto` | Numérico | Valor total contratado (R$) | Normalização Z-Score |
| `prazo_dias` | Numérico | Prazo contratual vigente (dias) | Normalização Z-Score |
| `num_aditivos` | Numérico | Quantidade de aditivos celebrados | Normalização Z-Score |
| `percentual_executado` | Numérico | Medição física acumulada (0 a 1) | Escala direta |
| `recurso_federal` | Binário | Indicador de fonte de repasse federal | Binário (0 ou 1) |
| `empresa_porte` | Categórico | Porte da empresa contratada | *Ordinal Encoding* |
| `indice_pluviometrico` | Numérico | Pluviosidade acumulada média (mm) | Normalização Z-Score |

> **Engenharia de Atributos:** Termos polinomiais de grau 2 e interações de primeira ordem gerados automaticamente pelo módulo `FeatureEngineer`.

---

## 4. Métricas de Avaliação e Desempenho

As métricas foram coletadas na base de teste estratificada (20% do volume total):

| Métrica | Algoritmo Selecionado (Random Forest) | Regressão Logística | Limiar de Aceitação (*Quality Gate*) |
| :--- | :--- | :--- | :--- |
| **Acurácia (Accuracy)** | ≥ 0.82 | ≥ 0.80 | ≥ 0.75 |
| **Precisão (Precision - Classe 1)** | ≥ 0.84 | ≥ 0.82 | ≥ 0.75 |
| **Revocação / Sensibilidade (Recall)** | ≥ 0.86 | ≥ 0.85 | ≥ 0.75 |
| **F1-Score (Macro)** | ≥ 0.82 | ≥ 0.81 | ≥ 0.75 |

---

## 5. Governança, Auditabilidade e Human-in-the-Loop

1. **Combate ao Desvio de Treino-Inferência (*Training-Serving Skew*):** A inferência online utiliza o `PredictionService`, aplicando exatamente os mesmos estimadores e alinhamentos de colunas (`feature_names_in_`) do treinamento.
2. **Critério de Promoção Automática (*Quality Gate*):** Apenas modelos com acurácia superior ao limiar configurado (`production_threshold >= 0.75`) são promovidos para `models/trained/production_model.pkl`.
3. **Detecção Contínua de Desvio (*Data Drift*):** A DAG do Airflow monitora deslocamentos na distribuição estatística das features ($> 0.5$ desvios padrão) emitindo alertas e registrando relatórios para auditoria antes de retreinamentos.
4. **Revisão Humana (*Human-in-the-Loop*):** As probabilidades emitidas pela API (`probability` e `confidence`) servem de índice orientativo para que auditores e engenheiros fundamentem suas decisões de campo.
