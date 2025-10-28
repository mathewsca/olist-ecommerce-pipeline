# 🧰 Tutorial do Makefile

O `Makefile` fornece uma interface simples e padronizada para as tarefas mais comuns do projeto. Assim, você não precisa lembrar comandos longos – basta usar `make <alvo>`.

## Pré‑requisitos

- Ter o `make` instalado
- Recomenda‑se usar um ambiente virtual Python ativo
- Alvos de Docker/Compose requerem Docker instalado; alvos de Airflow requerem Airflow instalado

## Como descobrir os comandos disponíveis

```bash
make help
```

## Fluxos comuns (mão na massa)

### 1) Preparar ambiente de desenvolvimento
```bash
make setup           # instala dependências de dev e configura pre-commit
# ou, mais básico
make install-dev     # instala deps de dev (sem mensagens finais)
make install         # apenas deps de runtime
```

### 2) Qualidade de código
```bash
make lint            # flake8 + mypy
make format          # black + isort (aplica correções)
make format-check    # black + isort em modo verificação (sem alterar arquivos)
make precommit       # roda hooks do pre-commit em todos os arquivos
```

### 3) Testes e cobertura
```bash
make test            # todos os testes com cobertura (HTML em htmlcov/)
make test-unit       # apenas unitários
make test-integration# apenas integração
```

### 4) Limpeza
```bash
make clean           # remove caches, builds, cobertura, etc.
```

### 5) API e Notebooks
```bash
make api             # inicia a API (Uvicorn, reload)
make jupyter         # inicia Jupyter Notebook na pasta notebooks/
make jupyter-lab     # inicia JupyterLab na pasta notebooks/
make notebook        # executa o notebook de exemplo
```

### 6) Docker/Compose
```bash
make docker-build    # build da imagem
make docker-run      # roda container (mapeia data/ e models/)
make docker-compose-up
make docker-compose-down
```

### 7) Airflow (local)
```bash
make airflow-init    # inicializa DB e cria usuário admin
make airflow-webserver
make airflow-scheduler
make airflow         # atalho: init + webserver
```

### 8) Pipelines e criação de projeto
```bash
make pipeline        # roda pipeline de exemplo
make create-project  # gera novo projeto via Cookiecutter
```

## Dicas rápidas

- Rode `make all` para uma verificação completa (clean, install-dev, test, lint).
- Após `make test`, abra `htmlcov/index.html` no navegador para ver a cobertura.
- Em macOS, você pode abrir a cobertura com `open htmlcov/index.html`.

## Problemas comuns

- "command not found: make": instale o `make` (em macOS, via Xcode CLT: `xcode-select --install` ou Homebrew `brew install make`).
- Erros de Docker/Airflow: verifique se as ferramentas estão instaladas e rodando.
- Erros de import: garanta que instalou com `pip install -e .` e está no diretório do projeto.
