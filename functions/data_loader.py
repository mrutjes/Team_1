import pandas as pd
import re

def data_loader(yields_path: str, smiles_path: str):
    # --- 1. Parser voor opbrengsten: haal hoogste percentage uit tekstregels ---
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
                    yield_data.append((compound_id, max_yield))

    df_yields_clean = pd.DataFrame(yield_data, columns=["compound_id", "yield"])

    # --- 2. Parser voor SMILES-bestand ---
    smiles_data = []
    with open(smiles_path, "r") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) == 4:
                compound_id, smiles_raw, borylation_index, _ = parts  # negeer genormaliseerde SMILES
                smiles_data.append((compound_id, smiles_raw, borylation_index))

    df_smiles_clean = pd.DataFrame(smiles_data, columns=["compound_id", "smiles_raw", "borylation_site"])

    # --- 3. Corrigeer 1-based borylation_site naar 0-based indexing ---
    df_smiles_clean["borylation_site"] = df_smiles_clean["borylation_site"].astype(int) - 1

    # --- Optioneel: waarschuwing bij negatieve indices ---
    if (df_smiles_clean["borylation_site"] < 0).any():
        print("⚠️ Waarschuwing: negatieve borylation indices gevonden na -1 correctie!")

    # --- 4. Merge datasets op compound_id ---
    df_merged = pd.merge(df_smiles_clean, df_yields_clean, on="compound_id", how="inner")

    df_merged["borylation_site"] = df_merged["borylation_site"].astype(int)
    df_merged["yield"] = df_merged["yield"].astype(float)
    return df_merged
