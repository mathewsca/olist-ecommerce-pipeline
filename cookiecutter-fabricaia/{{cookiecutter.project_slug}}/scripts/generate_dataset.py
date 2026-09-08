#!/usr/bin/env python3
"""
Gerador de dados sintéticos de contratos públicos para demonstração e teste end-to-end do template FabricaIA.
Gera dataset com indicadores de risco de atraso e paralisação de contratos públicos.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def generate_dataset(output_path: str = "data/raw/obras_publicas.csv", n_samples: int = 500, seed: int = 42):
    np.random.seed(seed)

    tipos = ["rodovia", "edificacao", "saneamento", "hospitalar", "educacao"]
    portes = ["pequeno", "medio", "grande"]

    data = {
        "id_obra": [f"OBR-{2024 + (i % 3):04d}-{i+1:04d}" for i in range(n_samples)],
        "tipo_obra": np.random.choice(tipos, n_samples),
        "valor_previsto": np.round(np.random.uniform(100000, 10000000, n_samples), 2),
        "prazo_dias": np.random.randint(60, 720, n_samples),
        "num_aditivos": np.random.poisson(lam=1.5, size=n_samples),
        "percentual_executado": np.round(np.random.uniform(0.05, 0.95, n_samples), 2),
        "recurso_federal": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        "empresa_porte": np.random.choice(portes, n_samples, p=[0.3, 0.5, 0.2]),
        "indice_pluviometrico": np.round(np.random.uniform(20.0, 350.0, n_samples), 1),
    }

    df = pd.DataFrame(data)

    # Regra de risco realista: aditivos altos, prazos longos, baixa execução e pequenas empresas aumentam o risco
    risk_score = (
        (df["num_aditivos"] * 0.4)
        + (df["prazo_dias"] / 365.0 * 0.3)
        - (df["percentual_executado"] * 1.5)
        + (df["empresa_porte"].map({"pequeno": 0.5, "medio": 0.1, "grande": -0.3}))
        + (df["indice_pluviometrico"] / 300.0 * 0.4)
        + np.random.normal(0, 0.3, n_samples)
    )

    df["atraso_risco"] = (risk_score > 0.3).astype(int)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Dataset gerado com sucesso em: {output_path} (Shape: {df.shape})")
    print("Distribuicao da variavel alvo (atraso_risco):")
    print(df["atraso_risco"].value_counts(normalize=True))
    return df


if __name__ == "__main__":
    generate_dataset()
