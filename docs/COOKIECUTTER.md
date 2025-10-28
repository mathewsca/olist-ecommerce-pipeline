# Como usar o Cookiecutter com este projeto

Este repositório inclui um template Cookiecutter em `cookiecutter-fabricaia/` para gerar novos projetos com a mesma estrutura do FabricaIA.

## Pré-requisitos

```bash
pip install cookiecutter
```

## Gerar um novo projeto (recomendado)

Gere o novo projeto como uma pasta-irmã (fora da raiz atual):

```bash
# Execute na raiz deste repositório
cookiecutter cookiecutter-fabricaia --output-dir ..
```

- Responda aos prompts (nome do projeto, autor, e opções como API, Airflow, Notebooks e Deep Learning).
- O projeto será criado em `../<project_slug>`.

## Estrutura gerada

- **Diretórios**: `data/`, `models/`, `pipelines/{dags,scripts}`, `src/{api,data,features,models,visualization}`, `tests/{unit,integration}`, `config/`, `notebooks/`, `docs/`.
- **Arquivos essenciais**: `pyproject.toml` (para `pip install -e .`), `requirements.txt`, `.gitignore`.

## Instalação no projeto gerado

```bash
cd ../<project_slug>

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
pip install -e .

# (Opcional) Kernel do Jupyter
python -m ipykernel install --user --name <project_slug>

# (Opcional) pre-commit
pip install pre-commit
pre-commit install
pre-commit run -a
```

## Opções do template

- **`include_api`**: se "n", remove `src/api/` (via hook pós-geração)
- **`include_airflow`**: se "n", remove `pipelines/dags/`
- **`include_notebooks`**: se "n", remove `notebooks/`
- **`include_deep_learning`**: controla dependências (não remove código; ajuste suas `requirements` conforme necessário)

## Modo não interativo (usa defaults)

```bash
cookiecutter cookiecutter-fabricaia --output-dir .. --no-input \
  project_name="Meu Projeto" \
  author_name="Seu Nome" \
  author_email="seu.email@exemplo.com" \
  include_deep_learning=y include_api=y include_airflow=y include_notebooks=y
```

## Alternativa: script de atalho

Você também pode usar o helper:

```bash
python create_project.py
```

Ele solicita os dados básicos e chama o Cookiecutter usando o template local.

## Solução de problemas

### "Gerou pasta vazia"
O Cookiecutter copia o conteúdo de `cookiecutter-fabricaia/{{cookiecutter.project_slug}}/`. Se essa pasta estiver vazia, o projeto gerado ficará vazio. Este repositório já inclui a estrutura mínima.

### "Quero gerar dentro da pasta atual"
Use sem `--output-dir` (não recomendado para evitar misturar gerado com o template).

### "Quero gerar em um caminho específico"
Use `--output-dir /caminho/destino`.

## Próximos passos

- Ajuste `config/config.yaml` para seu caso de uso.
- Crie seus pipelines em `pipelines/scripts/` e DAGs (se habilitado) em `pipelines/dags/`.
- Use a API (se habilitada) iniciando: `uvicorn src.api.main:app --reload`.
