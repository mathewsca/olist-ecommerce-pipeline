# FabricaIA Development Environment

## Quick Start Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov black flake8 isort mypy pre-commit
pip install -e .

# Setup pre-commit hooks
pre-commit install
pre-commit run -a
```

### Development
```bash
# Run tests
pytest tests/ -v

# Run linting
flake8 src/ tests/

# Format code
black src/ tests/

# Start API server
python -m uvicorn src.api.main:app --reload

# Start Jupyter Lab
jupyter lab notebooks/

# Run example pipeline
python pipelines/scripts/example_pipeline.py
```

### Docker
```bash
# Build image
docker build -t fabricaia .

# Run container
docker run -p 8000:8000 fabricaia

# Run with docker-compose
docker-compose up -d
```

### Airflow
```bash
# Initialize database
airflow db init

# Create admin user
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin

# Start webserver
airflow webserver --port 8080

# Start scheduler
airflow scheduler
```

## Project Structure

```
FabricaIA/
├── data/                    # Data storage
├── models/                  # Trained models
├── pipelines/               # ML pipelines
├── src/                     # Source code
│   ├── data/               # Data processing
│   ├── models/             # Model training
│   ├── features/           # Feature engineering
│   ├── visualization/      # Plotting
│   └── api/                # REST API
├── tests/                   # Test suite
├── config/                  # Configuration
├── notebooks/              # Jupyter notebooks
├── logs/                   # Log files
└── docs/                   # Documentation
```

## Key Features

- **Modular Design**: Reusable components for ML workflows
- **API Ready**: FastAPI for model inference
- **Pipeline Automation**: Apache Airflow integration
- **Docker Support**: Containerized deployment
- **Testing**: Comprehensive test suite
- **Documentation**: Jupyter notebooks with examples
- **Code Quality**: Pre-commit hooks and linting
