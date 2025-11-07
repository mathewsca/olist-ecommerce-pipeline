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

**Importante**: O Cookiecutter copia o conteúdo de `cookiecutter-fabricaia/{{cookiecutter.project_slug}}/` para criar o novo projeto. A estrutura gerada será:

```
<project_slug>/
├── config/
│   └── config.yaml
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
├── models/
│   ├── artifacts/
│   └── trained/
├── pipelines/
│   ├── dags/
│   └── scripts/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
│   ├── integration/
│   └── unit/
├── notebooks/
├── docs/
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

**Nota**: Alguns diretórios podem não ser criados se você escolher "n" para as opções correspondentes (ex: `src/api/` se `include_api=n`, `pipelines/dags/` se `include_airflow=n`, `notebooks/` se `include_notebooks=n`).

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

### "Gerou pasta vazia ou estrutura incompleta"
O Cookiecutter copia o conteúdo de `cookiecutter-fabricaia/{{cookiecutter.project_slug}}/`. Se essa pasta estiver vazia ou incompleta, o projeto gerado terá a mesma estrutura. Este repositório já inclui a estrutura mínima necessária.

**Verificação**: Confirme que o diretório `cookiecutter-fabricaia/{{cookiecutter.project_slug}}/` contém os arquivos e pastas esperados antes de gerar um novo projeto.

### "Quero gerar dentro da pasta atual"
Use sem `--output-dir` (não recomendado para evitar misturar gerado com o template).

### "Quero gerar em um caminho específico"
Use `--output-dir /caminho/destino`.

## Próximos passos

- Ajuste `config/config.yaml` para seu caso de uso.
- Crie seus pipelines em `pipelines/scripts/` e DAGs (se habilitado) em `pipelines/dags/`.
- Use a API (se habilitada) iniciando: `uvicorn src.api.main:app --reload`.
