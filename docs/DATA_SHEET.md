# Data Sheet — Ficha Técnica do Dataset de Contratos Públicos

Este documento adota o padrão *Datasheets for Datasets* (Gebru et al., 2021) para documentar a composição, proveniência, pré-processamento e limitações da base de dados utilizada no ciclo de vida de Machine Learning.

---

## 1. Motivação (Motivation)

* **Para qual finalidade o dataset foi criado?**  
  Para servir de base de referência e treinamento no estudo de caso de triagem de riscos em obras e serviços de engenharia no âmbito do template FabricaIA.
* **Quem financiou / criou o dataset?**  
  Equipe do projeto FabricaIA.
* **Comentários gerais:**  
  A base sintética foi calibrada com distribuições empíricas baseadas em contratações públicas para permitir testes reproduzíveis sem exposição de dados sensíveis ou sob sigilo.

---

## 2. Composição (Composition)

* **O que cada instância representa?**  
  Cada registro representa um contrato de obra pública individual com seus respectivos dados de acompanhamento físico-financeiro.
* **Quantidade de instâncias:**  
  500 contratos na versão padrão (gerados via `scripts/generate_dataset.py`).
* **Atributos:**
  1. `id_obra` (identificador textual, ex: `OBR-2024-0001`)
  2. `tipo_obra` (categoria: rodovia, saneamento, edificacao, hospitalar, educacao)
  3. `valor_previsto` (valor total contratado em reais)
  4. `prazo_dias` (prazo contratual em dias)
  5. `num_aditivos` (quantidade de termos aditivos celebrados)
  6. `percentual_executado` (percentual de execução física acumulado)
  7. `recurso_federal` (indicador booleano/inteiro: 0 ou 1)
  8. `empresa_porte` (porte empresarial da contratada: pequeno, medio, grande)
  9. `indice_pluviometrico` (precipitação média observada na localidade em mm)
  10. `atraso_risco` (variável alvo binária: 0 = regular, 1 = risco crítico de atraso/paralisação)
* **O dataset contém dados confidenciais ou de identificação pessoal (PII)?**  
  Não. Os dados são anonimizados e sintéticos.

---

## 3. Processo de Coleta e Geração (Collection Process)

* **Mecanismo de Geração:**  
  Script estocástico reprodutível com `random_seed=42` (`scripts/generate_dataset.py`).
* **Lógica de Risco Sintético:**  
  A probabilidade de risco (`atraso_risco=1`) é uma função logística ponderada pelo número de aditivos, defasagem entre prazo decorrido e avanço físico, porte da contratada e severidade de chuvas.

---

## 4. Pré-processamento e Feature Store (Preprocessing)

* **Limpeza e Imputação:** Executada por `DataProcessor.clean_data` (remoção de duplicatas e imputação de mediana/moda).
* **Transformação Categórica e Escalonamento:** Executada por `DataProcessor.encode_categorical` e `DataProcessor.scale_features` (com preservação estrita do rótulo `atraso_risco`).
* **Persistência em Feature Store:** Dados tratados e enriquecidos com interações polinomiais são persistidos em formato Apache Parquet colunar em `data/processed/feature_store.parquet`.

---

## 5. Limitações e Monitoramento de Data Drift

* **Cobertura Temporal e Geográfica:** A base padrão é representativa de contratos estaduais e municipais de médio porte. Para obras de infraestrutura complexa (ex: ferrovias, portos), recomenda-se calibração adicional.
* **Monitoramento Contínuo:** A tarefa `check_data_drift` na DAG do Airflow avalia rotineiramente o desvio normalizado de média ($\text{shift} > 0.5$) contra o baseline para sinalizar a necessidade de atualização desta ficha técnica e retreinamento do modelo.
