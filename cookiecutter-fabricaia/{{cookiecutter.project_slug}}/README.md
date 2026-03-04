# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Estrutura do Projeto

```text
{{ cookiecutter.project_slug }}/
├── config/             # Arquivos de configuração
├── data/               # Armazenamento de dados (raw, processed, external)
├── docs/               # Documentação técnica
├── models/             # Artefatos de modelos treinados
├── notebooks/          # Jupyter notebooks para experimentação
├── pipelines/          # Definições de pipelines e DAGs do Airflow
├── src/                # Código fonte do projeto (API, Processamento, Modelagem)
├── tests/              # Testes unitários e de integração
├── Makefile            # Comandos de automação
├── Dockerfile          # Configuração Docker
└── pyproject.toml      # Metadados e dependências do projeto
```

## Configuração do Ambiente

1.  **Pré-requisitos:** Python {{ cookiecutter.python_version }} ou superior e Docker (opcional).
2.  **Instalação:**
    ```bash
    make install
    ```
3.  **Ambiente de Desenvolvimento:**
    ```bash
    make setup
    ```

## Comandos Principais

-   `make test`: Executa a suíte de testes.
-   `make api`: Inicia o servidor da API FastAPI.
-   `make jupyter`: Inicia o ambiente Jupyter.
-   `make docker-compose-up`: Inicia todos os serviços via Docker Compose.

---
Gerado automaticamente a partir do template FabricaIA.
