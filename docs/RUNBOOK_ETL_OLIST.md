# Runbook de Execução  Projeto de Aula ETL (Olist)
## Guia Operacional Prático: Extract → Raw → Staging → Data Warehouse (Estrela) → Dashboard

Este documento descreve como executar, de ponta a ponta, o projeto de ETL construído sobre o template **FabricaIA** usando o *Brazilian E-Commerce Public Dataset by Olist* (Kaggle). Ele é independente do fluxo de MLOps descrito em [`RUNBOOK_END_TO_END.md`](RUNBOOK_END_TO_END.md).

> **Compatibilidade Multiplataforma:** cada passo mostra o comando via **`make`** e a alternativa **direta** (Docker/Python/psql), para funcionar tanto em Linux/macOS quanto no Windows (PowerShell).

---

## 1. O Estudo de Caso

Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) pedidos de um marketplace brasileiro entre 2016 e 2018. Usamos as 9 tabelas "core" do e-commerce (clientes, pedidos, itens, pagamentos, reviews, produtos, vendedores, geolocalização e tradução de categoria); o dataset separado de *marketing funnel/leads* **não é usado** neste projeto.

Objetivo pedagógico: percorrer um pipeline de ETL completo e realista extração de uma API externa, carga bruta, limpeza/tipagem, modelagem dimensional (esquema estrela) e consumo via dashboard usando Docker, Postgres, Airflow, Jupyter e Streamlit.

---

## 2. Visão Geral do Fluxo

```text
+------------------------------------------------------------------------------------------------+
|                              FLUXO ETL OLIST (schema estrela)                                    |
|                                                                                                  |
|  [Kaggle API / CSV manual] --> [raw.*] --> [staging.*] --> [dw.* fato+dimensao] --> [Streamlit]   |
|         make ingest          RawLoader   StagingTransformer      DWBuilder        dashboard      |
|                                                                                                  |
|                    Orquestrado ponta a ponta pela DAG `olist_etl_pipeline` (Airflow)              |
+------------------------------------------------------------------------------------------------+
```

---

## 3. Passo a Passo Operacional

### Passo 1 Configurar credenciais

Copie `.env.example` para `.env` e preencha:
- `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` — credenciais raiz do serviço `postgres` (metadata do Airflow). Já vêm com defaults (`airflow`/`airflow`/`airflow`) que funcionam sem alteração.
- `DW_*` — credenciais do Data Warehouse `olist_dw`. Por padrão reaproveitam o mesmo usuário raiz do Postgres; se você mudar `POSTGRES_USER`/`POSTGRES_PASSWORD`, mude `DW_USER`/`DW_PASSWORD` junto.
- `AIRFLOW_FERNET_KEY`/`AIRFLOW_JWT_SECRET` — segredos fixos para o Airflow (sem eles, cada restart de container gera novos e quebra as Connections salvas). Gere os seus com os comandos comentados no `.env.example`.
- `KAGGLE_USERNAME`/`KAGGLE_KEY` — opcional. Obtenha em [kaggle.com/settings](https://www.kaggle.com/settings) → API → *Create New Token*. Sem isso, a ingestão automática falha, mas o pipeline funciona normalmente se os CSVs forem colocados manualmente em `data/raw/`.

> [!NOTE]
> A senha do usuário `admin` da **UI do Airflow** não é configurada por variável de ambiente: o Airflow 3 gera uma senha aleatória a cada start do container `airflow-webserver` e a imprime no log. Para recuperar: `docker compose logs airflow-webserver | grep "Password for user"`.

```bash
cp .env.example .env
```

### Passo 2 Subir a infraestrutura

```bash
docker-compose up -d postgres airflow-init airflow-webserver airflow-scheduler jupyter streamlit
```

O serviço `airflow-init` roda `airflow db migrate` contra o Postgres (metadata do Airflow) e encerra sozinho; `postgres` cria automaticamente o banco `olist_dw` com os schemas `raw`/`staging`/`dw` vazios (via `postgres/init/01-init-olist-dw.sh`) na primeira inicialização do volume.

*Sem Docker Compose (execução nativa):*
```bash
make airflow-init
make airflow-webserver &
make airflow-scheduler &
```

### Passo 3 Registrar a conexão do Airflow com o Data Warehouse

A task `init_schemas` da DAG usa `PostgresHook`, que depende de uma Connection do Airflow chamada `olist_dw_postgres`.

```bash
make airflow-connections
```

*Ou via UI:* Admin → Connections → + → `Conn Id=olist_dw_postgres`, `Conn Type=Postgres`, `Host=postgres`, `Schema=olist_dw`, `Login=airflow`, `Password=airflow`, `Port=5432`.

### Passo 4 Obter o dataset (dois caminhos)

**Automático (via Kaggle API)** roda como a primeira task da DAG, ou manualmente:
```bash
make ingest
```

**Manual** baixe o dataset em [kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), extraia o zip e copie os 9 CSVs para `data/raw/`. A ingestão automática detecta os arquivos já presentes e pula o download os dois caminhos alimentam a mesma pasta.

### Passo 5 Rodar a DAG completa

Acesse http://localhost:8080 (usuário/senha padrão `admin`/`admin`, criados pelo `airflow-init` nativo no Docker Compose, crie via `docker-compose exec airflow-webserver airflow users create ...` se necessário), ative e dispare a DAG `olist_etl_pipeline`.

*Via CLI:*
```bash
docker-compose exec airflow-webserver airflow dags trigger olist_etl_pipeline
```

A DAG executa: `check_or_download_kaggle_data → init_schemas → load_raw (9 tabelas em paralelo) → build_staging_tables → build_dw_dimensions → build_dw_facts → validate_dw → cleanup_temp_files`.

### Passo 6 Validar o Data Warehouse

```bash
docker-compose exec postgres psql -U airflow -d olist_dw -c "SELECT COUNT(*) FROM dw.fact_order_items;"
```

Ou abra `notebooks/05_validacao_e_consultas_dw.ipynb` no Jupyter (http://localhost:8888) e rode o `DWValidator`.

### Passo 7 Explorar nos notebooks

Abra o Jupyter (http://localhost:8888) e siga a sequência:
1. `01_exploracao_dados_olist.ipynb` explorar os CSVs brutos.
2. `02_extract_kaggle_e_raw_load.ipynb` extração + carga raw.
3. `03_staging_limpeza.ipynb` limpeza/tipagem/deduplicação.
4. `04_modelagem_dw_star_schema.ipynb` construção do esquema estrela.
5. `05_validacao_e_consultas_dw.ipynb` validação + consultas de negócio.
6. `06_preview_dashboard.ipynb` prévia dos gráficos sem precisar do Streamlit.

### Passo 8 Abrir o dashboard

```bash
docker-compose up -d streamlit
```
Acesse http://localhost:8501. Três abas: **Dicionário de Dados** (modelo dimensional + amostra por tabela, com filtro de seleção), **Análise Exploratória** (distribuições, correlações, comparações) e **Dashboard** (receita, categorias, entregas, reviews, geografia e vendedores, cada seção já com um insight em linguagem de negócio).

*Sem Docker:*
```bash
make streamlit
```

### Passo 9 Testes

```bash
make etl-test           # unitarios (sem banco) - funcoes puras de staging_transform, kaggle_ingestion, db config
make test-integration  # integracao (usa testcontainers para subir um Postgres efemero)
```

---

## 4. Guia Rápido de Comandos (Cheatsheet)

| Objetivo | Com Makefile | Sem Makefile |
| :--- | :--- | :--- |
| **Baixar dataset do Kaggle** | `make ingest` | `python scripts/kaggle_ingest.py` |
| **Criar/recriar schemas do DW** | `make dw-init` | `psql $DATABASE_URL -f sql/raw/create_raw_schema.sql -f sql/staging/create_staging_schema.sql -f sql/dw/create_dw_schema.sql` |
| **Resetar o DW (destrutivo)** | `make dw-reset` | `psql ... -c "DROP SCHEMA dw, staging, raw CASCADE;"` seguido de `make dw-init` |
| **Registrar conexão Airflow→Postgres** | `make airflow-connections` | `airflow connections add olist_dw_postgres --conn-type postgres ...` |
| **Subir o dashboard** | `make streamlit` | `streamlit run src/dashboard/app.py` |
| **Testes unitários do ETL** | `make etl-test` | `pytest tests/unit/test_kaggle_ingestion.py tests/unit/test_staging_transform.py tests/unit/test_db_config.py -v` |
| **Testes de integração (DB real via testcontainers)** | `make test-integration` | `pytest tests/integration/ -v` |
| **Subir toda a stack** | `docker-compose up -d` |  |

---

## 5. Troubleshooting

| Sintoma | Causa Provável | Solução |
| :--- | :--- | :--- |
| `RuntimeError: KAGGLE_USERNAME/KAGGLE_KEY are not set` | Credenciais do Kaggle não configuradas e CSVs ausentes em `data/raw/` | Configure `.env` com as credenciais **ou** baixe o dataset manualmente e coloque os CSVs em `data/raw/` (Passo 4). |
| `401 - Unauthorized` na API do Kaggle | Token do Kaggle expirado/errado | Gere um novo token em kaggle.com/settings e atualize `KAGGLE_KEY` no `.env`. |
| `connection to server at "postgres" ... failed` no Streamlit/Jupyter | Serviço `postgres` ainda não terminou o healthcheck, ou DAG ainda não populou o DW | Aguarde o `postgres` subir e rode a DAG `olist_etl_pipeline` antes de abrir o dashboard. |
| DAG falha em `init_schemas` com `connection "olist_dw_postgres" not found` | Connection do Airflow não registrada | Rode `make airflow-connections` ou cadastre manualmente na UI (Passo 3). |
| `relation "dw.dim_customers" does not exist` | Tentou popular fatos/dimensões sem antes rodar `init_schemas`/`DDL` | Rode a DAG completa (que já faz `init_schemas` primeiro) ou `make dw-init` manualmente. |
| Airflow ainda aponta para SQLite (metadata) | `airflow-init` não rodou, ou `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` não chegou ao container | Confirme que `docker-compose up airflow-init` rodou com sucesso antes do webserver/scheduler. |
| `make test` falha em máquina sem Docker | `tests/integration/` precisa de `testcontainers`, que sobe um Postgres via Docker | Rode `make test-unit` (não depende de Docker); use `make test-integration` só onde Docker está disponível. |

---

## 6. Ver também

- [`RUNBOOK_END_TO_END.md`](RUNBOOK_END_TO_END.md)  fluxo de MLOps original do template (não relacionado a este projeto de ETL).
- [`ESTRUTURA_PROJETO.md`](ESTRUTURA_PROJETO.md) árvore de diretórios do projeto, incluindo os novos módulos de ETL.
