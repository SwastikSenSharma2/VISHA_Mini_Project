import pandas as pd, json, os

# Path to the antivenom Excel file copied into the project root
excel_path = r'C:\\Users\\Swastik Sen Sharma\\Downloads\\VISHA-main\\SBE-profiles.xlsx'

try:
    df = pd.read_excel(excel_path)
    print('Loaded Excel with shape:', df.shape)
    # Assume columns: 'Species' and 'Antivenom' (adjust if names differ)
    species_col = None
    antivenom_col = None
    for col in df.columns:
        lowered = str(col).lower()
        if 'species' in lowered:
            species_col = col
        if 'antivenom' in lowered or 'product' in lowered:
            antivenom_col = col
    if species_col is None or antivenom_col is None:
        raise ValueError('Could not locate species or antivenom columns in Excel')

    mapping = dict(zip(df[species_col].astype(str).str.strip(), df[antivenom_col].astype(str).str.strip()))
    # Write mapping to a Python module for import
    out_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'antivenom_lookup.py')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('"""Auto‑generated antivenom lookup dictionary from SBE‑profiles.xlsx"""\n')
        f.write('ANTIVENOM_MAP = {\n')
        for k, v in mapping.items():
            f.write(f"    {repr(k)}: {repr(v)},\n")
        f.write('}\n')
    print('Antivenom lookup module written to', out_path)
except Exception as e:
    print('Error processing Excel:', e)
