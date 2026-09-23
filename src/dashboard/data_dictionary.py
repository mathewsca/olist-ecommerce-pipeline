"""
FabricaIA - Data Dictionary

Static metadata describing the star-schema Data Warehouse (dw.*): table
grain/purpose and column descriptions. Used by the "Dicionário de Dados" tab
so students can read what each table/column means without opening the SQL
DDL in sql/dw/create_dw_schema.sql.
"""

DW_TABLES = {
    "dim_customers": {
        "kind": "Dimensão",
        "description": "Um registro por cliente do marketplace, com a localização (CEP/cidade/estado/região).",
        "columns": {
            "customer_key": "Chave substituta (surrogate key), gerada pelo DW.",
            "customer_id": "Chave natural do Olist - identifica o cliente em um pedido específico.",
            "customer_unique_id": "Identificador estável do cliente entre vários pedidos.",
            "customer_zip_code_prefix": "Prefixo do CEP (5 dígitos, com zeros à esquerda restaurados).",
            "customer_city": "Cidade padronizada (minúscula, sem acento, reparada pelo CEP quando inválida).",
            "customer_state": "UF validada contra as 27 UFs.",
            "customer_region": "Macro-região (Norte, Nordeste, Centro-Oeste, Sudeste, Sul).",
            "zip_in_geolocation": "Qualidade: o CEP existe em dim_geolocation (dá para plotar no mapa).",
            "zip_state_mismatch": "Qualidade: a UF diverge da faixa de CEP dos Correios.",
        },
    },
    "dim_sellers": {
        "kind": "Dimensão",
        "description": "Um registro por vendedor (seller) do marketplace.",
        "columns": {
            "seller_key": "Chave substituta, gerada pelo DW.",
            "seller_id": "Chave natural do Olist.",
            "seller_zip_code_prefix": "Prefixo do CEP (5 dígitos).",
            "seller_city": "Cidade padronizada (e-mails, números e siglas no campo cidade foram reparados).",
            "seller_state": "UF validada.",
            "seller_region": "Macro-região.",
            "zip_in_geolocation": "Qualidade: o CEP existe em dim_geolocation.",
            "zip_state_mismatch": "Qualidade: a UF diverge da faixa de CEP dos Correios.",
        },
    },
    "dim_products": {
        "kind": "Dimensão",
        "description": (
            "Um registro por produto, com categoria (traduzida e agrupada em macro-categorias) "
            "e dimensões físicas."
        ),
        "columns": {
            "product_key": "Chave substituta, gerada pelo DW.",
            "product_id": "Chave natural do Olist.",
            "product_category_name": "Categoria em português (snake_case); 'unknown' quando o produto não tem categoria.",
            "product_category_name_english": "Categoria traduzida (inclui as 2 categorias que faltavam na tabela do Kaggle).",
            "category_group": "Macro-categoria de negócio (Casa e Decoração, Moda, Eletrônicos...).",
            "product_weight_g": "Peso em gramas (zeros viraram nulos e foram imputados).",
            "product_length_cm": "Comprimento em cm.",
            "product_height_cm": "Altura em cm.",
            "product_width_cm": "Largura em cm.",
            "product_volume_cm3": "Volume = comprimento x altura x largura.",
            "product_photos_qty": "Quantidade de fotos no anúncio (nulo = anúncio sem metadados).",
            "product_name_length": "Tamanho do título do anúncio (a coluna original tinha o erro 'lenght').",
            "product_description_length": "Tamanho da descrição do anúncio.",
            "has_listing_metadata": "Qualidade: FALSE quando fotos/título/descrição não foram informados.",
            "dimensions_imputed": "Qualidade: peso/dimensões foram preenchidos com a mediana da categoria.",
        },
    },
    "dim_geolocation": {
        "kind": "Dimensão",
        "description": (
            "Um registro por prefixo de CEP, com latitude/longitude média. O raw tem ~1 milhão de pontos "
            "(261 mil duplicados exatos e alguns fora do Brasil): foram limpos e agregados na staging."
        ),
        "columns": {
            "geo_key": "Chave substituta, gerada pelo DW.",
            "zip_code_prefix": "Prefixo do CEP (5 dígitos).",
            "city": "Cidade mais frequente do CEP, já padronizada.",
            "state": "UF mais frequente do CEP.",
            "region": "Macro-região.",
            "avg_lat": "Latitude média dos pontos válidos do CEP.",
            "avg_lng": "Longitude média dos pontos válidos do CEP.",
            "n_points": "Quantos pontos brutos entraram na média.",
            "n_points_discarded": "Quantos pontos foram descartados como outliers de GPS.",
        },
    },
    "dim_date": {
        "kind": "Dimensão",
        "description": "Uma linha por dia do calendário, cobrindo o intervalo de datas de pedidos e avaliações.",
        "columns": {
            "date_key": "Chave no formato YYYYMMDD, usada como FK pelas fatos.",
            "full_date": "Data completa (DATE).",
            "year": "Ano.",
            "quarter": "Trimestre (1-4).",
            "month": "Mês (1-12).",
            "day": "Dia do mês.",
            "day_of_week": "Dia da semana (0=segunda .. 6=domingo).",
            "is_weekend": "Verdadeiro se sábado ou domingo.",
            "year_month": "Rótulo ordenável 'AAAA-MM'.",
            "month_name": "Nome do mês em português.",
            "day_name": "Nome do dia da semana em português.",
            "week_of_year": "Semana ISO do ano.",
        },
    },
    "fact_order_items": {
        "kind": "Fato",
        "grain": "Uma linha por item de pedido (grão transacional).",
        "description": "Preço e frete de cada item vendido, ligado a cliente/vendedor/produto/data.",
        "columns": {
            "order_item_key": "Chave substituta, gerada pelo DW.",
            "order_id": "Identificador do pedido (natural key do Olist).",
            "order_item_id": "Número sequencial do item dentro do pedido.",
            "customer_key": "FK para dim_customers.",
            "seller_key": "FK para dim_sellers.",
            "product_key": "FK para dim_products.",
            "order_purchase_date_key": "FK para dim_date (data da compra).",
            "price": "Preço do item (R$), sempre > 0.",
            "freight_value": "Valor do frete do item (R$), sempre >= 0.",
            "is_price_outlier": "Qualidade: preço acima de Q3 + 3*IQR (mantido, mas sinalizado).",
            "is_freight_above_price": "Qualidade: frete maior que o preço do item.",
        },
    },
    "fact_orders": {
        "kind": "Fato",
        "grain": "Uma linha por pedido (grão de pedido, mais grosso que fact_order_items).",
        "description": "Status, tempos do ciclo do pedido e totais, calculados por pedido.",
        "columns": {
            "order_key": "Chave substituta, gerada pelo DW.",
            "order_id": "Chave natural do Olist (única).",
            "customer_key": "FK para dim_customers.",
            "order_status": "Status do pedido (delivered, shipped, canceled...).",
            "order_purchase_date_key": "FK para dim_date (data da compra).",
            "order_delivered_date_key": "FK para dim_date (data da entrega ao cliente).",
            "order_estimated_delivery_date_key": "FK para dim_date (prazo estimado).",
            "purchase_hour": "Hora da compra (0-23).",
            "approval_time_hours": "Horas entre a compra e a aprovação do pagamento.",
            "carrier_handoff_days": "Dias entre a compra e a entrega à transportadora.",
            "delivery_time_days": "Dias entre compra e entrega (só pedidos entregues e com linha do tempo coerente).",
            "delivery_delay_days": "Dias de atraso vs. prazo estimado, por data (negativo = adiantado).",
            "is_late": "Verdadeiro se entregue depois da data estimada.",
            "has_date_anomaly": "Qualidade: datas fora de ordem ou impossíveis (ver staging.dq_issues).",
            "items_count": "Quantidade de itens do pedido.",
            "items_value": "Soma dos preços dos itens (R$).",
            "freight_value": "Soma dos fretes (R$).",
        },
    },
    "fact_payments": {
        "kind": "Fato",
        "grain": "Uma linha por registro de pagamento (um pedido pode ter mais de um).",
        "description": "Forma e valor de pagamento de cada pedido.",
        "columns": {
            "payment_key": "Chave substituta, gerada pelo DW.",
            "order_id": "Identificador do pedido.",
            "payment_sequential": "Sequencial do pagamento dentro do pedido.",
            "payment_type": "Tipo de pagamento (credit_card, boleto, voucher, debit_card, not_defined).",
            "payment_installments": "Número de parcelas (mínimo 1; zeros foram corrigidos).",
            "payment_value": "Valor pago (R$).",
            "is_valid_payment": "Qualidade: FALSE para tipo 'not_defined' ou valor ausente.",
        },
    },
    "fact_reviews": {
        "kind": "Fato",
        "grain": "Uma linha por avaliação de pedido (o mesmo review_id pode aparecer em mais de um pedido).",
        "description": "Nota, comentário limpo e metadados da avaliação deixada pelo cliente.",
        "columns": {
            "review_key": "Chave substituta, gerada pelo DW.",
            "review_id": "Identificador da avaliação - NÃO é único sozinho; a chave é (review_id, order_id).",
            "order_id": "Identificador do pedido avaliado.",
            "review_score": "Nota de 1 a 5.",
            "review_creation_date_key": "FK para dim_date (data da avaliação).",
            "review_answer_date_key": "FK para dim_date (data da resposta).",
            "has_comment_title": "Se o cliente escreveu um título de comentário.",
            "has_comment_message": "Se o cliente escreveu uma mensagem de comentário.",
            "review_comment_title": "Título sanitizado (sem quebras de linha nem espaços duplos).",
            "review_comment_message": "Comentário sanitizado; caixa e acentos preservados.",
            "comment_length": "Número de caracteres do comentário.",
            "is_low_information": "Qualidade: comentário de até 3 caracteres ou só pontuação ('ok', '.', '10').",
            "is_shouting": "Comentário todo em maiúsculas.",
            "is_latest_for_order": "Verdadeiro na avaliação mais recente do pedido (evita contar 2x).",
            "response_time_hours": "Horas entre a criação da avaliação e a resposta.",
        },
    },
}

# Auditoria da camada staging (não fazem parte do esquema estrela).
QUALITY_TABLES = {
    "staging.dq_issues": (
        "Uma linha por regra de qualidade avaliada: tabela, coluna, regra, ação tomada "
        "(fixed/flagged/dropped/imputed/nullified/kept), linhas afetadas, % e exemplos antes → depois."
    ),
    "staging.dq_column_profile": (
        "Perfil por coluna antes (raw) e depois (staging): tipo, nulos e valores distintos."
    ),
}

STAR_SCHEMA_OVERVIEW = """
**Modelo dimensional (esquema estrela)** - camada `dw` do Data Warehouse Olist.

- **Dimensões** (`dim_*`): descrevem quem/o que/onde - clientes, vendedores, produtos, geografia e datas.
- **Fatos** (`fact_*`): medem eventos de negócio, cada um com um grão explícito.
  - `fact_order_items` - grão de **item de pedido** (transacional, mais fino).
  - `fact_orders` - grão de **pedido** (inclui métricas de entrega).
  - `fact_payments` - grão de **pagamento**.
  - `fact_reviews` - grão de **avaliação**.

Colunas marcadas como **Qualidade** são flags geradas na limpeza: os registros suspeitos **não são apagados**,
são sinalizados, para que cada análise decida se os inclui ou não. O que foi corrigido em cada tabela está na
aba **Análise Exploratória → Qualidade dos dados**.

Use o filtro abaixo para explorar a estrutura e ver uma amostra de cada tabela.
"""
