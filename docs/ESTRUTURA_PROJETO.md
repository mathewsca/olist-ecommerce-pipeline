# Estrutura do Projeto FabricaIA

## 📁 Diretórios Principais

| Diretório | Descrição |
|-----------|-----------|
| `data/` | Armazenamento de dados do projeto |
| ├── `raw/` | Dados brutos, não processados |
| ├── `processed/` | Dados processados e limpos |
| └── `external/` | Dados externos ou de terceiros |
| `models/` | Modelos de ML treinados |
| ├── `trained/` | Modelos salvos (arquivos .pkl) |
| └── `artifacts/` | Artefatos do modelo (métricas, gráficos) |
| `pipelines/` | Pipelines de automação |
| ├── `dags/` | DAGs do Apache Airflow (inclui `olist_etl_pipeline_dag.py`) |
| └── `scripts/` | Scripts de pipeline Python |
| `src/` | Código fonte do projeto |
| ├── `data/` | Módulos de processamento de dados (ML) |
| ├── `etl/` | Módulos de ETL do Data Warehouse Olist (extract, raw, staging, dw, validações) |
| ├── `dashboard/` | Dashboard Streamlit sobre o Data Warehouse (`app.py` + `tabs/`, 3 abas) |
| ├── `models/` | Módulos de treinamento de modelos |
| ├── `features/` | Módulos de engenharia de features |
| ├── `visualization/` | Módulos de visualização |
| └── `api/` | API REST com FastAPI |
| `sql/` | DDL do Data Warehouse (`raw/`, `staging/`, `dw/`) |
| `postgres/init/` | Script de inicialização do banco `olist_dw` no primeiro start do container |
| `tests/` | Testes automatizados |
| ├── `unit/` | Testes unitários |
| └── `integration/` | Testes de integração (Postgres via testcontainers) |
| `config/` | Arquivos de configuração |
| `notebooks/` | Notebooks Jupyter (inclui a sequência `01`–`06` do projeto de ETL Olist) |
| `docs/` | Documentação do projeto |
| `logs/` | Arquivos de log |

## 📄 Arquivos de Configuração

### Principais
- **`requirements.txt`** - Dependências Python (sklearn, pandas, matplotlib, numpy, FastAPI, Airflow, etc.)
- **`Dockerfile`** - Container Docker com ambiente completo
- **`docker-compose.yml`** - Orquestração de serviços (API, Jupyter, Airflow, PostgreSQL)
- **`.gitignore`** - Arquivos ignorados pelo Git
- **`README.md`** - Documentação principal do projeto

### Desenvolvimento
- **`.pre-commit-config.yaml`** - Hooks de pré-commit (Black, Flake8, isort, mypy)
- **`pytest.ini`** - Configuração de testes
- **`Makefile`** - Comandos úteis para desenvolvimento
- **`setup.py`** - Configuração do pacote Python
- **`DEVELOPMENT.md`** - Guia de desenvolvimento


### Cookiecutter
- **`cookiecutter-fabricaia/`** - Template para novos projetos
- **`create_project.py`** - Script para criar novos projetos

## 🔧 Módulos de Código Reutilizável

### 1. Data Processing (`src/data/processor.py`)
```python
from src.data.processor import DataProcessor

processor = DataProcessor()
df = processor.load_data('data.csv')
df_clean = processor.clean_data(df)
df_encoded = processor.encode_categorical(df_clean)
df_scaled = processor.scale_features(df_encoded)
X_train, X_test, y_train, y_test = processor.split_data(df_scaled, 'target')
```

**Funcionalidades:**
- Carregamento de dados (CSV, Excel, JSON, Parquet)
- Limpeza de dados (valores faltantes, duplicados)
- Codificação de variáveis categóricas
- Normalização/escala de features
- Divisão treino/teste

### 2. Model Training (`src/models/trainer.py`)
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(model_type='classification')
model = trainer.train_model(X_train, y_train, 'random_forest')
metrics = trainer.evaluate_model(model, X_test, y_test)
cv_results = trainer.cross_validate(model, X_train, y_train)
trainer.save_model(model, 'models/trained/model.pkl')
```

**Funcionalidades:**
- Treinamento de modelos (Random Forest, Logistic Regression, SVM)
- Avaliação de modelos
- Validação cruzada
- Otimização de hiperparâmetros (GridSearchCV)
- Persistência de modelos

### 3. Feature Engineering (`src/features/engineering.py`)
```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer()
df_poly = engineer.create_polynomial_features(df, ['feature1', 'feature2'])
df_interact = engineer.create_interaction_features(df, [('f1', 'f2')])
df_binned = engineer.create_binning_features(df, ['age'], bins=5)
df_selected = engineer.select_features(X, y, method='univariate', k=10)
```

**Funcionalidades:**
- Features polinomiais
- Features de interação
- Rolling features (séries temporais)
- Lag features
- Binning
- Seleção de features (Univariate, RFE, PCA)
- Features estatísticas

### 4. Visualization (`src/visualization/plots.py`)
```python
from src.visualization.plots import DataVisualizer

visualizer = DataVisualizer()
visualizer.plot_distribution(df, ['feature1', 'feature2'])
visualizer.plot_correlation_matrix(df)
visualizer.plot_feature_importance(importance_df)
visualizer.plot_model_performance(metrics, 'Random Forest')
visualizer.plot_prediction_vs_actual(y_test, y_pred)
```

**Funcionalidades:**
- Distribuições de variáveis
- Matriz de correlação
- Importância de features
- Performance de modelos
- Predições vs valores reais
- Séries temporais
- Dados faltantes

## 🚀 Pipelines e Automação

### Pipeline de Exemplo (`pipelines/scripts/example_pipeline.py`)
Pipeline completo demonstrando:
1. Carregamento e exploração de dados
2. Pré-processamento
3. Engenharia de features
4. Treinamento de múltiplos modelos
5. Avaliação e comparação
6. Salvamento de modelos

### DAG do Airflow (`pipelines/dags/fabricaia_pipeline_dag.py`)
Pipeline automatizado com tarefas:
- `load_and_explore_data`
- `preprocess_data`
- `engineer_features`
- `train_models`
- `evaluate_models`
- `cleanup_temp_files`

### 5. ETL do Data Warehouse Olist (`src/etl/`)
```python
from src.etl.db import get_engine
from src.etl.raw_loader import RawLoader
from src.etl.staging_transform import StagingTransformer
from src.etl.dw_builder import DWBuilder
from src.etl.validations import DWValidator

engine = get_engine()
RawLoader(engine).load_all()
StagingTransformer(engine).build_all()
builder = DWBuilder(engine)
builder.build_all_dimensions()
builder.build_all_facts()
DWValidator(engine).run_all()
```

**Funcionalidades:**
- `kaggle_ingestion.py` — download idempotente do dataset via Kaggle API
- `raw_loader.py` — carga dos CSVs no schema `raw`
- `staging_transform.py` — limpeza/tipagem/deduplicação
- `dw_builder.py` — construção do esquema estrela (dimensões + fatos) no schema `dw`
- `validations.py` — contagens mínimas e integridade de chaves estrangeiras

Orquestrado pela DAG `olist_etl_pipeline` (`pipelines/dags/olist_etl_pipeline_dag.py`) e documentado em [`RUNBOOK_ETL_OLIST.md`](RUNBOOK_ETL_OLIST.md).

### 6. Dashboard (`src/dashboard/`)
App Streamlit (`app.py` + `tabs/`) lendo o schema `dw` via `src/etl/db.py`, organizado em três abas:
- **Dicionário de Dados** (`tabs/dictionary_tab.py` + `data_dictionary.py`) — modelo dimensional, descrição de colunas e amostra de 5 linhas por tabela (filtro de seleção).
- **Análise Exploratória** (`tabs/exploratory_tab.py`) — distribuições, matriz de correlação, comparações entre grupos (categoria, estado).
- **Dashboard** (`tabs/insights_tab.py`) — receita, categorias, entregas, reviews, geografia e vendedores, cada seção com um insight em linguagem de negócio.

## 🌐 API REST

### FastAPI (`src/api/main.py`)

**Endpoints disponíveis:**
- `GET /` - Informações da API
- `GET /health` - Health check
- `GET /models` - Listar modelos disponíveis
- `POST /predict` - Fazer predição única
- `POST /predict_batch` - Predição em lote (CSV)
- `GET /model/{model_name}/info` - Informações do modelo

**Exemplo de uso:**
```bash
# Iniciar API
uvicorn src.api.main:app --reload

# Fazer predição
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 30, "income": 50000}, "model_name": "random_forest"}'
```

## 🧪 Testes

### Suite de Testes (`tests/unit/test_fabricaia.py`)

**Cobertura:**
- ✅ DataProcessor (limpeza, encoding, scaling, split)
- ✅ ModelTrainer (treinamento, avaliação, cross-validation, persistência)
- ✅ FeatureEngineer (polynomial, interaction, binning, selection)
- ✅ DataVisualizer (plots básicos)
- ✅ Teste de integração (pipeline completo)

**Executar testes:**
```bash
pytest tests/ -v --cov=src
```

## 📊 Notebooks

### `notebooks/fabricaia_example.ipynb`
Notebook completo demonstrando:
1. Importação de bibliotecas
2. Exploração de dados
3. Visualizações
4. Pré-processamento
5. Engenharia de features
6. Treinamento de modelos
7. Avaliação e comparação
8. Otimização de hiperparâmetros
9. Salvamento de modelos

## 🐳 Docker

### Dockerfile
- Base: Python 3.11-slim
- Bibliotecas: sklearn, pandas, matplotlib, numpy, FastAPI, Airflow
- Porta exposta: 8000
- Comando padrão: API server

### Docker Compose
Serviços configurados:
- **fabricaia-api** (porta 8000)
- **jupyter** (porta 8888)
- **airflow-init** (roda `airflow db migrate` e encerra)
- **airflow-webserver** (porta 8080)
- **airflow-scheduler**
- **streamlit** (porta 8501) — dashboard do Data Warehouse Olist
- **postgres** (porta 5432) — bancos `airflow` (metadata) e `olist_dw` (Data Warehouse)

## 📋 Makefile - Comandos Úteis

```bash
make help              # Mostrar ajuda
make install           # Instalar dependências
make install-dev       # Instalar dev dependencies
make test              # Executar testes
make lint              # Executar linting
make format            # Formatar código
make clean             # Limpar arquivos temporários
make docker-build      # Build Docker image
make docker-compose-up # Iniciar todos os serviços
make api               # Iniciar API
make jupyter           # Iniciar Jupyter
make airflow-init      # Inicializar Airflow
make pipeline          # Executar pipeline exemplo
make ingest            # Baixar dataset Olist do Kaggle (ou pular se já existir)
make dw-init           # Criar schemas/tabelas do Data Warehouse
make streamlit         # Iniciar o dashboard
make etl-test          # Testes unitários do ETL
```

## 🎯 Próximos Passos

1. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Executar testes:**
   ```bash
   pytest tests/ -v
   ```

3. **Explorar o notebook de exemplo:**
   ```bash
   jupyter notebook notebooks/fabricaia_example.ipynb
   ```

4. **Iniciar a API:**
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```

5. **Usar Docker:**
   ```bash
   docker-compose up -d
   ```

## 📚 Documentação Adicional

- `README.md` - Visão geral e guia de uso
- `DEVELOPMENT.md` - Guia de desenvolvimento
- `RUNBOOK_END_TO_END.md` - Fluxo de MLOps original do template
- `RUNBOOK_ETL_OLIST.md` - Fluxo do projeto de ETL (Olist): extract → raw → staging → DW estrela → dashboard
- `config/config.yaml` - Configurações do projeto
- Notebooks em `notebooks/` - Exemplos práticos
