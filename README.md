# Fábrica de Inteligência Artificial do Piauí

**Acelerando o desenvolvimento de soluções baseadas em Inteligência Artificial para o Governo do Estado do Piauí.**

Este repositório abriga os artefatos técnicos, metodológicos e científicos do projeto **Fábrica de Inteligência Artificial (IA)**, uma iniciativa da **Secretaria de Inteligência Artificial, Economia Digital, Ciência e Tecnologia, Inovação e Transformação Digital (SIA/PI)** em parceria com a **Fundação de Amparo à Pesquisa do Estado do Piauí (FAPEPI)** e a **Universidade Federal do Piauí (UFPI)**.

---

## Sumário
- [Visão Geral](#-visão-geral)
- [Objetivos](#-objetivos)
- [Características](#-características-do-repositório)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Instalação](#️-instalação)
- [Uso Rápido](#-uso-rápido)
- [Fluxo de Execução End-to-End](#-fluxo-de-execução-end-to-end)
- [Configuração](#-configuração)
- [MLflow Tracking](#-mlflow-tracking)
- [API REST](#-api-rest)
- [Pipelines com Airflow](#-pipelines-com-airflow)
- [Testes](#-testes)
- [Notebooks](#-notebooks)
- [Contribuição](#-contribuição)

---

## Visão Geral

A **Fábrica de IA** é um centro estratégico voltado ao desenvolvimento de soluções **preditivas e generativas de Inteligência Artificial**, com o objetivo de otimizar processos, reduzir custos e aprimorar a tomada de decisão no governo estadual.

O projeto está inserido no contexto da **plataforma SoberanIA**, iniciativa do Governo do Piauí para consolidar uma infraestrutura de IA em português, baseada em um supercomputador de alta performance.

---

## Objetivos

### Objetivo Geral
Desenvolver e implementar uma Fábrica de Inteligência Artificial para o Governo do Estado do Piauí, criando soluções inovadoras baseadas em IA preditiva e generativa e promovendo a integração entre **governo, universidade e sociedade**.

### Objetivos Específicos
1. **Desenvolver Modelos de IA:** criar e aperfeiçoar quatro modelos de IA aplicados à gestão pública.
2. **Promover a Formação de Talentos:** capacitar servidores, pesquisadores e profissionais locais.
3. **Estimular a Pesquisa Aplicada:** integrar IA à pesquisa em parceria com universidades.
4. **Estruturar Arquitetura Escalável:** disponibilizar soluções via APIs seguras e integráveis.
5. **Monitorar Impactos:** definir métricas e indicadores de eficiência e impacto social.
6. **Consolidar o Piauí como Referência em IA:** promover o estado como polo nacional de inovação pública.


## 🚀 Características do Repositório

- **Modular**: Componentes reutilizáveis para processamento de dados, treinamento de modelos e visualização
- **Escalável**: Suporte para pipelines automatizados com Apache Airflow
- **API Ready**: Interface REST para inferência de modelos com FastAPI
- **Dockerizado**: Ambiente containerizado com todas as dependências necessárias
- **Testado**: Suite completa de testes unitários e de integração
- **Documentado**: Notebooks Jupyter com exemplos práticos

> Dica: consulte o tutorial do Makefile em `docs/MAKEFILE.md` para conhecer os atalhos de desenvolvimento (instalação, testes, lint, API, Docker, Airflow, etc.).

## 📁 Estrutura do Projeto

```
FabricaIA/
├── data/                          # Dados do projeto
│   ├── raw/                       # Dados brutos
│   ├── processed/                 # Dados processados
│   └── external/                  # Dados externos
├── models/                        # Modelos treinados
│   ├── trained/                   # Modelos salvos
│   └── artifacts/                # Artefatos do modelo
├── pipelines/                     # Pipelines de ML
│   ├── dags/                      # DAGs do Airflow
│   └── scripts/                   # Scripts de pipeline
├── scripts/                       # Scripts utilitários e geradores de dados
├── src/                          # Código fonte
│   ├── data/                     # Processamento de dados
│   ├── models/                   # Treinamento de modelos
│   ├── features/                 # Engenharia de features
│   ├── services/                 # Serviços de inferência (prevenção de skew)
│   ├── visualization/            # Visualizações
│   └── api/                      # API REST
├── tests/                        # Testes
│   ├── unit/                     # Testes unitários
│   └── integration/              # Testes de integração
├── config/                       # Configurações
├── notebooks/                   # Notebooks Jupyter
├── docs/                        # Documentação
├── logs/                        # Logs do sistema
├── Dockerfile                   # Container Docker
├── requirements.txt             # Dependências Python
├── .gitignore                   # Arquivos ignorados pelo Git
└── README.md                    # Este arquivo
```

## 🛠️ Instalação

### Pré-requisitos

- Python 3.11+
- Docker (opcional)

### Instalação Local

1. Clone o repositório:
```bash
git clone https://github.com/fabricaIA/FabricaIA.git
cd FabricaIA
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências e o pacote em modo editável:
```bash
pip install -r requirements.txt
pip install -e .
```

4. (Opcional) Configure o kernel do Jupyter apontando para o seu venv:
```bash
python -m ipykernel install --user --name fabricaia
```

5. (Opcional, recomendado) Configure o pre-commit:
```bash
pip install pre-commit
pre-commit install           # instala o hook no repositório
pre-commit run -a            # executa em todos os arquivos
```

### Instalação com Docker

```bash
# Build da imagem
docker build -t fabricaia .

# Executar container
docker run -p 8000:8000 fabricaia
```

## 🚀 Uso Rápido

### 1. Processamento de Dados

```python
from src.data.processor import DataProcessor

# Inicializar processador
processor = DataProcessor()

# Carregar dados
df = processor.load_data('data/raw/your_data.csv')

# Limpar dados
df_clean = processor.clean_data(df, drop_duplicates=True)

# Codificar variáveis categóricas
df_encoded = processor.encode_categorical(df_clean)

# Escalar features
df_scaled = processor.scale_features(df_encoded)
```

### 2. Treinamento de Modelos

```python
from src.models.trainer import ModelTrainer

# Inicializar trainer (com MLflow habilitado por padrão)
trainer = ModelTrainer(model_type='classification')

# Dividir dados
X_train, X_test, y_train, y_test = processor.split_data(df_scaled, 'target')

# Treinar modelo com tracking MLflow
model = trainer.train_model(
    X_train,
    y_train,
    'random_forest',
    run_name='random_forest_baseline',
    n_estimators=100,
    max_depth=10
)

# Avaliar modelo (métricas são automaticamente logadas no MLflow)
metrics = trainer.evaluate_model(model, X_test, y_test)

# Salvar modelo (também salva no MLflow)
trainer.save_model(model, 'models/trained/model.pkl', artifact_path='random_forest_model')

# Encerrar run do MLflow
trainer.end_run()
```

### 3. Engenharia de Features

```python
from src.features.engineering import FeatureEngineer

# Inicializar engenheiro de features
engineer = FeatureEngineer()

# Criar features polinomiais
df_poly = engineer.create_polynomial_features(df, ['feature1', 'feature2'])

# Criar features de interação
df_interact = engineer.create_interaction_features(df_poly, [('feature1', 'feature2')])
```

### 4. Visualização

```python
from src.visualization.plots import DataVisualizer

# Inicializar visualizador
visualizer = DataVisualizer()

# Plotar distribuições
visualizer.plot_distribution(df, ['feature1', 'feature2'])

# Matriz de correlação
visualizer.plot_correlation_matrix(df)

# Importância das features
visualizer.plot_feature_importance(importance_df)
```

## 🔧 Configuração

O arquivo `config/config.yaml` contém todas as configurações do projeto:

```yaml
# Caminhos dos dados
DATA_PATHS:
  raw: "data/raw"
  processed: "data/processed"
  external: "data/external"

# Configurações do pipeline
PIPELINE:
  test_size: 0.2
  random_state: 42
  cv_folds: 5

# Algoritmos disponíveis
MODELS:
  algorithms:
    - "random_forest"
    - "logistic_regression"
    - "svm"
```

## 📊 MLflow Tracking

O projeto inclui integração completa com MLflow para tracking de experimentos:

```bash
# Iniciar interface MLflow
make mlflow-ui

# Ou iniciar servidor MLflow completo
make mlflow-server
```

Acesse a interface em: `http://localhost:5001`

### Usando MLflow no código

```python
from src.models.trainer import ModelTrainer
from src.models.mlflow_tracker import MLflowTracker

# O trainer já vem com MLflow habilitado por padrão
trainer = ModelTrainer(model_type='classification')

# Cria um run customizado
tracker = MLflowTracker(experiment_name='meu_experimento')
run = tracker.start_run(run_name='teste_modelo_v1', tags={'version': '1.0'})

# Train model - automaticamente loga no MLflow
model = trainer.train_model(X_train, y_train, 'random_forest', run_name='rf_v1')

# Loga métricas
metrics = trainer.evaluate_model(model, X_test, y_test)

# Fecha o run
tracker.end_run()
```

## 🌐 API REST

A API REST permite inferência de modelos via HTTP:

```bash
# Iniciar API
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Fazer predição
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": {"feature1": 1.0, "feature2": 2.0}, "model_name": "random_forest"}'
```

## 🔄 Pipelines com Airflow

Exemplo de DAG para automação:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator

# Definir DAG
dag = DAG('fabricaia_pipeline', schedule_interval='@daily')

# Tarefas do pipeline
load_data = PythonOperator(task_id='load_data', python_callable=load_data_func)
preprocess = PythonOperator(task_id='preprocess', python_callable=preprocess_func)
train_model = PythonOperator(task_id='train_model', python_callable=train_model_func)

# Definir dependências
load_data >> preprocess >> train_model
```

## 🔁 Fluxo de Execução End-to-End

O template permite executar e reproduzir o ciclo completo de desenvolvimento de ML e MLOps em poucos passos integrados. Para um guia operacional aprofundado com payloads, comandos e troubleshooting, consulte o [Runbook de Execução End-to-End](docs/RUNBOOK_END_TO_END.md).

### 1. Geração da Base de Dados
Gere uma base de dados sintética realista para experimentação (ou insira seu arquivo em `data/raw/`):
```bash
make data
# Ou: python scripts/generate_dataset.py
# Gera data/raw/obras_publicas.csv (500 contratos com atributos numéricos, categóricos e target de risco)
```

### 2. Rastreamento com MLflow (Local ou Remoto)
Inicie o servidor MLflow local na porta 5001:
```bash
make mlflow-server &
```
> Acesse: `http://localhost:5001`

Para servidores remotos protegidos por autenticação HTTP Basic, basta configurar as variáveis de ambiente (ou definir em `config/config.yaml`):
```bash
export MLFLOW_TRACKING_URI="http://servidor-remoto:5000"
export MLFLOW_TRACKING_USERNAME="seu_usuario"
export MLFLOW_TRACKING_PASSWORD="sua_senha"
```

### 3. Pipeline de Treinamento e Serialização
Execute o pipeline unificado de ponta a ponta:
```bash
make pipeline
# Ou: python pipelines/scripts/example_pipeline.py
```
O pipeline executa:
- Carregamento e análise exploratória (com geração de gráficos em `logs/`)
- Limpeza, imputação e normalização com `DataProcessor`
- Geração de atributos polinomiais e de interação com `FeatureEngineer`
- Treinamento comparativo de modelos (**Random Forest**, **Regressão Logística** e **SVM**) com validação cruzada
- Registro de hiperparâmetros, métricas (Acurácia, F1-Score, Precisão, Revocação) e matrizes de confusão no MLflow
- Serialização dos modelos treinados em `models/trained/`

### 4. Serving com FastAPI (Prevenção de Training-Serving Skew)
Inicie a API REST com live-reload:
```bash
make api
# Acesse a documentação interativa Swagger UI em: http://localhost:8000/docs
```

A API utiliza a camada de serviço `PredictionService` (`src/services/prediction_service.py`), garantindo que os dados de inferência passem exatamente pelo mesmo pipeline de transformação do treino:

* **Inferência por Lookup de Instância (Zero Skew):**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"lookup_id": "OBR-2024-0001", "model_name": "random_forest"}'
```

* **Inferência com Atributos Brutos:**
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
         "percentual_executado": 0.4,
         "recurso_federal": 1,
         "empresa_porte": "grande",
         "indice_pluviometrico": 120.0
       }
     }'
```

* **Inferência em Lote via Planilha CSV:**
```bash
curl -X POST "http://localhost:8000/predict_batch?model_name=random_forest" \
     -F "file=@data/raw/obras_publicas.csv"
```

### 5. Orquestração Contínua com Apache Airflow
A DAG em `pipelines/dags/fabricaia_pipeline_dag.py` orquestra o ciclo contínuo de MLOps:
- Ingestão e validação
- Verificação de **Data Drift** (mudança na distribuição dos dados de entrada)
- Pré-processamento e persistência na **Feature Store** (`data/processed/feature_store.parquet`)
- Retreinamento e avaliação
- **Quality Gate & Promoção:** Se o modelo superar o limiar estipulado (ex: acurácia $\ge 0.75$), ele é promovido automaticamente como o modelo oficial de produção (`models/trained/production_model.pkl`).


## 🧪 Testes

Execute a suite de testes (assegure-se de ter instalado com `pip install -e .` ou exporte `PYTHONPATH=.`):

```bash
# Testes unitários
pytest tests/unit/

# Testes de integração
pytest tests/integration/

# Todos os testes
pytest tests/
```

## 📊 Notebooks

Explore os notebooks Jupyter em `notebooks/`:

- `fabricaia_example.ipynb`: Exemplo completo de uso do framework
- `data_exploration.ipynb`: Análise exploratória de dados
- `model_comparison.ipynb`: Comparação de modelos

## 🤝 Contribuição

1. Clone o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request
