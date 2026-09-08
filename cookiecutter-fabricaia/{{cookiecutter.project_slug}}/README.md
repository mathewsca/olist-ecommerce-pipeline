# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Estrutura do Projeto

```text
{{ cookiecutter.project_slug }}/
├── config/             # Arquivos de configuração centralizada
├── data/               # Armazenamento de dados (raw, processed, external)
├── docs/               # Documentação técnica e governança (Model Card, Data Sheet)
├── models/             # Artefatos e modelos treinados serializados
├── notebooks/          # Jupyter notebooks para experimentação
├── pipelines/          # Definições de pipelines e DAGs do Airflow
├── scripts/            # Scripts utilitários (ex: geração de dados sintéticos)
├── src/                # Código fonte do projeto
│   ├── api/            # API REST com FastAPI
│   ├── data/           # Processamento e limpeza de dados
│   ├── features/       # Engenharia de atributos
│   ├── models/         # Treinamento, avaliação e tracking MLflow
│   ├── services/       # Camada de serviços de inferência (prevenção de skew)
│   └── visualization/  # Diagnósticos visuais e matrizes de correlação
├── tests/              # Testes automatizados (pytest e cobertura)
├── Makefile            # Comandos de automação do ciclo de vida
├── Dockerfile          # Configuração Docker
└── pyproject.toml      # Metadados e dependências do projeto
```

## Configuração do Ambiente

1. **Pré-requisitos:** Python {{ cookiecutter.python_version }} ou superior e Docker (opcional).
2. **Instalação das Dependências:**
   ```bash
   make install   # Instalação padrão (dependências e pacote -e . sem hooks de pre-commit)
   # Ou para setup completo de desenvolvimento (incluindo linters e hooks de pre-commit):
   make setup
   ```

## 🔁 Fluxo de Execução End-to-End

O projeto vem pronto para reproduzir o ciclo completo de ML e MLOps. Para o guia detalhado com comandos, payloads cURL e troubleshooting, consulte o [Runbook de Execução End-to-End](docs/RUNBOOK_END_TO_END.md).

### 1. Geração de Dados
```bash
make data
# Gera base sintética data/raw/obras_publicas.csv pronta para experimentação
```

### 2. Tracking com MLflow (Local ou Remoto)
```bash
# Inicia servidor MLflow local na porta 5001
make mlflow-server &
```
> Acesse: `http://localhost:5001`

*Para MLflow remoto com autenticação HTTP Basic:*
```bash
export MLFLOW_TRACKING_URI="http://servidor-remoto:5000"
export MLFLOW_TRACKING_USERNAME="seu_usuario"
export MLFLOW_TRACKING_PASSWORD="sua_senha"
```

### 3. Pipeline de Treinamento
```bash
make pipeline
# Executa EDA, pré-processamento, engenharia de features, treino (Random Forest, Logistic Regression, SVM),
# log de métricas/artefatos no MLflow e serialização em models/trained/
```

### 4. Serving via API REST (FastAPI)
```bash
make api
# Acesse a documentação interativa Swagger UI em: http://localhost:8000/docs
```
* Exemplos de requisição:
  * **Lookup de Instância (Zero Skew):** `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"lookup_id": "OBR-2024-0001", "model_name": "random_forest"}'`
  * **Atributos Brutos:** `curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"model_name": "random_forest", "features": {"tipo_obra": "rodovia", "valor_previsto": 1500000.0, "prazo_dias": 365, "num_aditivos": 2, "percentual_executado": 0.4, "recurso_federal": 1, "empresa_porte": "grande", "indice_pluviometrico": 120.0}}'`
  * **Inferência em Lote:** `curl -X POST "http://localhost:8000/predict_batch?model_name=random_forest" -F "file=@data/raw/obras_publicas.csv"`

### 5. Qualidade de Software e Linters
```bash
make test         # Executa suite de testes unitários com cobertura
make lint         # Análise estática com flake8 e mypy
make format       # Formatação com black e isort
```

---
Gerado automaticamente a partir do template FabricaIA.
