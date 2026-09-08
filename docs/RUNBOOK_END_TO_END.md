# Runbook de Execução End-to-End — FabricaIA Template
## Guia Operacional Prático para Desenvolvimento, Serving e MLOps

Este documento fornece um guia passo a passo prático para experimentar, executar e reproduzir o ciclo de vida completo entregue pelo template **FabricaIA**, desde o scaffolding inicial e engenharia de dados até o serving em produção com combate ao *training-serving skew* e orquestração de MLOps.

---

## 1. O Estudo de Caso: Predição de Risco em Contratos Públicos (ObraAlerta)

Para demonstrar os recursos do template com dados estruturados realistas, utilizamos como estudo de caso a **predição de risco de atraso e paralisação de contratos públicos**:
* **Alvo (`atraso_risco`):** Variável binária (`0` = Em dia / `1` = Alto risco de paralisação ou atraso crítico).
* **Entradas:** Prazos, aditivos contratuais, medição física de execução, porte da empresa e indicadores ambientais/pluviométricos.
* **Objetivo de Negócio:** Permitir a triagem antecipada e a priorização de vistorias técnicas e auditorias preventivas.

---

## 2. Visão Geral do Ciclo de Vida

```text
+-----------------------------------------------------------------------------------------+
|                                  FLUXO END-TO-END                                       |
|                                                                                         |
|  [Scaffolding] ──> [Geração de Dados] ──> [Treino & MLflow] ──> [Serving REST FastAPI]   |
|         │                  │                      │                     │               |
|   Cookiecutter      make data (CSV)         make pipeline        make api (Anti-Skew)   |
|                                                   │                     │               |
|                                           [Airflow MLOps] <─────────────┘               |
|                                          (Drift, Store, Gate)                           |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Passo a Passo Operacional

### Passo 1 — Scaffolding e Configuração do Projeto

Para instanciar um novo projeto derivado do template:
```bash
# Executar assistente interativo de scaffolding
python create_project.py
```
*Ou via linha de comando sem prompt:*
```bash
cookiecutter cookiecutter-fabricaia --no-input \
  project_name="obra_alerta" \
  author_name="Equipe IA" \
  author_email="ia@exemplo.gov.br" \
  python_version="3.11"
```

Acesse o diretório do projeto e configure o ambiente de desenvolvimento:
```bash
cd obra_alerta

# 1. Crie e ative o ambiente virtual:
python3 -m venv venv
source venv/bin/activate   # Linux/macOS (ou venv\Scripts\activate no Windows)

# 2. Instale as dependências:
# Opção A.1 — Instalação padrão (sem configurar hooks de pre-commit):
make install

# Opção A.2 — Setup de desenvolvimento completo (com linters e hooks de pre-commit):
make setup

# Opção B — Diretamente via pip (universal para Linux, macOS e Windows sem make):
pip install -r requirements.txt
pip install -e .
```
> [!TIP]
> Se desejar apenas executar o fluxo sem ativar os hooks de validação do git (`pre-commit`), utilize **`make install`** em vez de `make setup`. O `make install` instala todas as bibliotecas do projeto, o suporte a testes com `pytest` e registra o pacote em modo editável (`-e .`).

Inspecione o arquivo central de configuração `config/config.yaml`:
```yaml
DATA_PATHS:
  raw: "data/raw"
  processed: "data/processed"
PIPELINE:
  test_size: 0.2
  random_state: 42
MODELS:
  algorithms:
    - "random_forest"
    - "logistic_regression"
    - "svm"
MLFLOW:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "fabricaia_experiments"
  enable_tracking: true
```

---

### Passo 2 — Geração da Base de Dados

Gere uma base sintética pronta para experimentação:
```bash
make data
```
*Ou execute diretamente:*
```bash
python scripts/generate_dataset.py
```
O arquivo é gerado em `data/raw/obras_publicas.csv` com 500 contratos e as seguintes colunas:
* `id_obra`: Identificador do contrato (ex: `OBR-2024-0001`).
* `tipo_obra`: Categoria da intervenção (`rodovia`, `edificacao`, `saneamento`, `hospitalar`, `educacao`).
* `valor_previsto`: Valor total contratual em R$.
* `prazo_dias`: Duração estipulada em dias.
* `num_aditivos`: Quantidade de termos aditivos de prazo/valor.
* `percentual_executado`: Medição física acumulada (0.05 a 0.95).
* `recurso_federal`: Indicador se recebe repasse federal (0 ou 1).
* `empresa_porte`: Porte da contratada (`pequeno`, `medio`, `grande`).
* `indice_pluviometrico`: Média de chuvas acumulada no período (mm).
* `atraso_risco`: Rótulo de risco (0 ou 1).

---

### Passo 3 — Experiment Tracking com MLflow

#### Opção A — Servidor Local (SQLite na porta 5001)
```bash
make mlflow-server &
```
> Acesse no navegador: `http://localhost:5001`  
*(A porta 5001 é utilizada por padrão para evitar conflito com portas reservadas do sistema operacional).*

#### Opção B — Servidor Remoto (com Autenticação HTTP Basic)
Defina as variáveis de ambiente antes de executar o pipeline:
```bash
export MLFLOW_TRACKING_URI="http://servidor-remoto:5000"
export MLFLOW_TRACKING_USERNAME="seu_usuario"
export MLFLOW_TRACKING_PASSWORD="sua_senha"
```
*As credenciais também podem ser definidas diretamente na seção `MLFLOW:` do `config/config.yaml`.*

---

### Passo 4 — Execução do Pipeline de Treinamento

Execute o pipeline consolidado:
```bash
make pipeline
```
*Ou:*
```bash
python pipelines/scripts/example_pipeline.py
```

O pipeline executa de forma integrada:
1. **Diagnóstico Exploratório:** Gera gráficos de distribuição e correlação salvos em `logs/`.
2. **Tratamento de Dados (`DataProcessor`):** Imputação de nulos, encoding ordinal de categóricas e normalização z-score das variáveis explicativas numéricas (preservando o target discreto).
3. **Engenharia de Atributos (`FeatureEngineer`):** Criação de variáveis polinomiais e de interação.
4. **Treinamento e Validação Cruzada (`ModelTrainer`):** Ajuste dos modelos **Random Forest**, **Regressão Logística** e **SVM**.
5. **Rastreabilidade e Artefatos:** Registro de parâmetros, acurácia, F1-Score, matrizes de confusão e modelos serializados em `models/trained/*.pkl` e no MLflow.

---

### Passo 5 — Serving REST e Combate ao Training-Serving Skew

Inicie o servidor de inferência com live-reload:
```bash
make api
```
> Documentação interativa (Swagger UI): `http://localhost:8000/docs`

#### Prevenção de Training-Serving Skew (`PredictionService`)
Em sistemas produtivos, dados enviados na inferência chegam frequentemente em formato bruto ou apenas como um identificador de registro. A camada `PredictionService` (`src/services/prediction_service.py`) intercepta a chamada e reaplica exatamente as mesmas regras de transformação e alinhamento do treinamento:

#### A. Inferência por Lookup de Instância (Zero Skew)
O cliente informa apenas o identificador da obra. A API busca o registro bruto e executa todo o pré-processamento antes de consultar o modelo:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "lookup_id": "OBR-2024-0001",
    "model_name": "random_forest"
  }'
```
*Resposta esperada (JSON):*
```json
{
  "prediction": 1,
  "probability": {
    "0": 0.27,
    "1": 0.73
  },
  "model_name": "random_forest",
  "confidence": 0.73,
  "lookup_id": "OBR-2024-0001",
  "raw_input_summary": {
    "id_obra": "OBR-2024-0001",
    "tipo_obra": "hospitalar",
    "valor_previsto": 5101998.49,
    "prazo_dias": 440,
    "num_aditivos": 5,
    "percentual_executado": 0.41,
    "empresa_porte": "grande"
  }
}
```

#### B. Inferência com Atributos Brutos
Permite submeter os dados de um contrato novo diretamente no payload JSON:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "random_forest",
    "features": {
      "tipo_obra": "rodovia",
      "valor_previsto": 1500000.0,
      "prazo_dias": 365,
      "num_aditivos": 2,
      "percentual_executado": 0.40,
      "recurso_federal": 1,
      "empresa_porte": "grande",
      "indice_pluviometrico": 120.0
    }
  }'
```

#### C. Inferência em Lote via Planilha CSV
Para classificar grandes volumes de contratos em lote:
```bash
curl -X POST "http://localhost:8000/predict_batch?model_name=random_forest" \
  -F "file=@data/raw/obras_publicas.csv"
```

---

### Passo 6 — Orquestração Contínua de MLOps com Apache Airflow

O arquivo `pipelines/dags/fabricaia_pipeline_dag.py` modela o ciclo contínuo de produção:

```text
load_and_explore_data
        ↓
check_data_drift          (Compara lote atual com histórico e avalia desvio estatístico)
        ↓
preprocess_data & engineer_features
        ↓
sync_feature_store        (Exporta dados limpos para data/processed/feature_store.parquet)
        ↓
train_models & evaluate_models
        ↓
promote_production_model  (Quality Gate: se acurácia >= 0.75, promove para models/trained/production_model.pkl)
        ↓
cleanup_temp_files
```

Para inicializar e subir o ambiente Airflow:
```bash
make airflow-init
make airflow-webserver     # No Airflow 3 executa api-server; no Airflow 2 executa webserver
# Em outro terminal:
make airflow-scheduler
```
> Acesse a interface web em `http://localhost:8080` (usuário: `admin` / senha: `admin`).

*Dica:* Para executar todos os serviços do Airflow (banco, scheduler e api-server) juntos em um único terminal:
```bash
make airflow-standalone
```

Para encerrar os processos do Airflow quando terminar:
```bash
make airflow-stop
```

---

### Passo 7 — Qualidade de Software, Testes e Linters

Execute a suíte automatizada com medição de cobertura de código:
```bash
make test
```
*Ou via pytest direto:*
```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
```

Validação estática e formatação:
```bash
make lint         # Análise estática com flake8 e mypy
make format       # Formatação com black e isort
make precommit    # Execução de todos os hooks do pre-commit
```

---

### Passo 8 — Containerização com Docker e Docker Compose

Construa e execute a imagem padronizada do projeto:
```bash
make docker-build
make docker-run
```

Para orquestrar múltiplos serviços simultaneamente (API FastAPI, Jupyter e Airflow):
```bash
docker-compose up -d
docker-compose ps
docker-compose down
```

---

### Passo 9 — Governança e Auditoria Técnica

O template disponibiliza templates de governança prontos em `docs/`:
1. **`docs/MODEL_CARD.md`:** Documenta a finalidade pretendida do modelo, limitações de uso, métricas de avaliação, dependências de hardware/software e considerações éticas (*human-in-the-loop*).
2. **`docs/DATA_SHEET.md`:** Registra a proveniência dos dados, procedimentos de amostragem, esquema de atributos e limitações de cobertura temporal ou geográfica.

---

## 4. Guia Rápido de Comandos (Cheatsheet)

| Objetivo | Com Makefile | Comando Direto (sem Makefile / Windows) |
| :--- | :--- | :--- |
| **Instalação Padrão (sem pre-commit)** | `make install` | `pip install -r requirements.txt && pip install -e .` |
| **Setup Completo (com pre-commit)** | `make setup` | `pip install -r requirements.txt && pip install -e . && pre-commit install` |
| **Gerar Base Sintética** | `make data` | `python scripts/generate_dataset.py` |
| **MLflow Server Local** | `make mlflow-server &` | `mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db &` |
| **Pipeline Completo de ML** | `make pipeline` | `python pipelines/scripts/example_pipeline.py` |
| **API REST FastAPI** | `make api` | `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload` |
| **Suite de Testes** | `make test` | `pytest tests/ -v --cov=src` |
| **Linters Estáticos** | `make lint` | `flake8 src/ tests/ && mypy src/` |
| **Formatação de Código** | `make format` | `black src/ tests/ && isort src/ tests/` |
| **Build Docker** | `make docker-build` | `docker build -t fabricaia .` |
| **Subir Serviços Docker**| `docker-compose up -d` | `docker-compose up -d` |

---

## 5. Troubleshooting (Resolução de Problemas Comuns)

| Sintoma | Causa Provável | Solução |
| :--- | :--- | :--- |
| `Command 'make' not found` | Utilitário `make` não instalado no sistema | No Linux (Ubuntu/Debian): `sudo apt install -y make`. Alternativamente, use os comandos diretos de `python`/`pip` listados na tabela acima (sem necessidade de instalar `make`). |
| `Address already in use: 5000` | Porta 5000 ocupada por outro serviço | O Makefile e a configuração usam a porta **5001**: `make mlflow-server`. |
| `ModuleNotFoundError: No module named 'src.services'` | Pacote não instalado no modo editável | Execute `pip install -e .` na raiz do projeto. |
| `IndexError: single positional indexer out-of-bounds` | Dataset sem colunas categóricas | Corrigido em `DataProcessor.clean_data` com verificação prévia de existência de colunas. |
| `Changing param values is not allowed` no MLflow | Runs concorrentes não finalizadas | Assegure que cada modelo feche explicitamente a run com `tracker.end_run()`. |
| `ValueError: Unknown label type: continuous` | Normalização aplicada sobre o target | O pipeline exclui automaticamente a coluna target da etapa `scale_features`. |
