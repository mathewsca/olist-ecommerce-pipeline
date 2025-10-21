# Fábrica de Inteligência Artificial do Piauí

**Acelerando o desenvolvimento de soluções baseadas em Inteligência Artificial para o Governo do Estado do Piauí.**

Este repositório abriga os artefatos técnicos, metodológicos e científicos do projeto **Fábrica de Inteligência Artificial (IA)**, uma iniciativa da **Secretaria de Inteligência Artificial, Economia Digital, Ciência e Tecnologia, Inovação e Transformação Digital (SIA/PI)** em parceria com a **Fundação de Amparo à Pesquisa do Estado do Piauí (FAPEPI)** e a **Universidade Federal do Piauí (UFPI)**.

---

## Sumário
- [Visão Geral](#-visão-geral)
- [Objetivos](#-objetivos)

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
├── src/                          # Código fonte
│   ├── data/                     # Processamento de dados
│   ├── models/                   # Treinamento de modelos
│   ├── features/                 # Engenharia de features
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
git clone <repository-url>
cd FabricaIA
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
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

# Inicializar trainer
trainer = ModelTrainer(model_type='classification')

# Dividir dados
X_train, X_test, y_train, y_test = processor.split_data(df_scaled, 'target')

# Treinar modelo
model = trainer.train_model(X_train, y_train, 'random_forest')

# Avaliar modelo
metrics = trainer.evaluate_model(model, X_test, y_test)
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

## 🧪 Testes

Execute a suite de testes:

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

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🆘 Suporte

Para suporte e dúvidas:

- Abra uma [issue](https://github.com/your-repo/FabricaIA/issues)
- Consulte a [documentação](docs/)
- Entre em contato: [seu-email@exemplo.com]

## 🎯 Roadmap

- [ ] Suporte para deep learning (TensorFlow, PyTorch)
- [ ] Integração com MLflow para experimentos
- [ ] Interface web para visualização
- [ ] Suporte para dados de streaming
- [ ] Integração com cloud providers (AWS, GCP, Azure)

---

**FabricaIA** - Construindo o futuro do Machine Learning, um módulo por vez! 🚀