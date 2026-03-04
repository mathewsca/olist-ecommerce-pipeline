# 🚀 FabricaIA - Guia de Início Rápido

## Instalação em 3 Passos

### 1️⃣ Criar Ambiente Virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 2️⃣ Instalar Dependências
```bash
pip install -r requirements.txt
pip install -e .
```

### 3️⃣ Verificar Instalação
```bash
pytest tests/ -v
```

### (Opcional) Jupyter Kernel usando seu venv
```bash
python -m ipykernel install --user --name fabricaia
```

### (Opcional, recomendado) Pre-commit
```bash
pip install pre-commit
pre-commit install           # instala o hook no repositório
pre-commit run -a            # executa em todos os arquivos
```

## 🧰 Usando o Makefile (atalhos úteis)

Os comandos mais comuns do projeto estão no `Makefile`.
Veja o tutorial completo em `docs/MAKEFILE.md`.

```bash
make help          # lista todos os alvos disponíveis
make setup         # instala deps de dev + pre-commit
make test          # roda testes com cobertura
make lint          # flake8 + mypy
make format        # black + isort (aplica)
make format-check  # black + isort (verificação)
make api           # inicia a API com reload
```

## 📊 Uso Básico - Exemplo Completo

### Pipeline Simples de ML

```python
import pandas as pd
import numpy as np
from src.data.processor import DataProcessor
from src.models.trainer import ModelTrainer
from src.visualization.plots import DataVisualizer

# 1. Criar dados de exemplo
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'education_years': np.random.randint(8, 20, 1000),
    'target': np.random.randint(0, 2, 1000)
})

# 2. Processar dados
processor = DataProcessor()
df_clean = processor.clean_data(data, drop_duplicates=True)
df_scaled = processor.scale_features(df_clean)

# 3. Dividir dados
X_train, X_test, y_train, y_test = processor.split_data(df_scaled, 'target')

# 4. Treinar modelo
trainer = ModelTrainer(model_type='classification')
model = trainer.train_model(X_train, y_train, 'random_forest')

# 5. Avaliar modelo
metrics = trainer.evaluate_model(model, X_test, y_test)
print(f"Acurácia: {metrics['accuracy']:.4f}")

# 6. Salvar modelo
trainer.save_model(model, 'models/trained/my_model.pkl')

# 7. Visualizar resultados
visualizer = DataVisualizer()
visualizer.plot_model_performance(metrics, 'Random Forest')
```

## 🔥 Comandos Mais Usados

### Desenvolvimento
```bash
# Executar pipeline de exemplo
python pipelines/scripts/example_pipeline.py

# Iniciar Jupyter Notebook
jupyter notebook notebooks/fabricaia_example.ipynb

# Formatar código
black src/ tests/

# Executar testes
pytest tests/ -v --cov=src
```

### API REST
```bash
# Iniciar servidor
python -m uvicorn src.api.main:app --reload

# Testar API (em outro terminal)
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/models
```

### Docker
```bash
# Build e executar
docker build -t fabricaia .
docker run -p 8000:8000 fabricaia

# Ou usar docker-compose
docker-compose up -d
```

### Airflow
```bash
# Inicializar
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# Iniciar serviços
airflow webserver --port 8080  # Terminal 1
airflow scheduler              # Terminal 2
```

## 📁 Arquivos Importantes

- **`requirements.txt`** - Todas as dependências
- **`config/config.yaml`** - Configurações do projeto
- **`Dockerfile`** - Container Docker
- **`Makefile`** - Comandos úteis (veja `docs/MAKEFILE.md`)
- **`notebooks/fabricaia_example.ipynb`** - Exemplo completo

## 🎯 Casos de Uso

### 1. Processamento de Dados
```python
from src.data.processor import DataProcessor

processor = DataProcessor()
df = processor.load_data('data/raw/dataset.csv')
df_clean = processor.clean_data(df, handle_missing='fill')
df_encoded = processor.encode_categorical(df_clean)
```

### 2. Engenharia de Features
```python
from src.features.engineering import FeatureEngineer

engineer = FeatureEngineer()
df_poly = engineer.create_polynomial_features(df, ['age', 'income'])
df_interact = engineer.create_interaction_features(df_poly, [('age', 'income')])
```

### 3. Treinamento de Modelos
```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(model_type='classification')
model = trainer.train_model(X_train, y_train, 'random_forest')
metrics = trainer.evaluate_model(model, X_test, y_test)
```

### 4. Visualização
```python
from src.visualization.plots import DataVisualizer

visualizer = DataVisualizer()
visualizer.plot_distribution(df, ['age', 'income'])
visualizer.plot_correlation_matrix(df)
```

## 🐛 Solução de Problemas

### Erro: "ModuleNotFoundError"
```bash
# Certifique-se de estar no diretório correto
cd FabricaIA

# Ative o ambiente virtual
source venv/bin/activate

# Reinstale as dependências
pip install -r requirements.txt
```

### Erro: "PYTHONPATH not set"
```bash
# Linux/Mac
export PYTHONPATH=FabricaIA:$PYTHONPATH

# Windows
set PYTHONPATH=C:\path\to\FabricaIA;%PYTHONPATH%
```

### Erro: "Port already in use"
```bash
# Encontrar processo usando a porta
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Matar processo
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```
