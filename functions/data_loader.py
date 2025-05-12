import pandas as pd
import re

def data_loader(yields_path: str, smiles_path: str):
    
    # --- 2. Parser voor yields: hoogste percentage extraheren ---
    yield_data = []
    with open(yields_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                compound_id, yield_info = parts
                percentages = re.findall(r'(\d+)%', yield_info)
                if percentages:
                    max_yield = max(map(int, percentages))
                    yield_data.append((compound_id, int(max_yield)))

    df_yields_clean = pd.DataFrame(yield_data, columns=["compound_id", "yield"])

    # --- 3. Parser voor SMILES ---
    smiles_data = []
    with open(smiles_path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) == 4:
                compound_id, smiles_raw, number, _ = parts  # ignore smiles_normalized
                smiles_data.append((compound_id, smiles_raw, number))

    df_smiles_clean = pd.DataFrame(
        smiles_data,
        columns=["compound_id", "smiles_raw", "borylation_site"]
    )

    # --- 4. Merge op compound_id ---
    return pd.merge(df_smiles_clean, df_yields_clean, on="compound_id", how="inner")