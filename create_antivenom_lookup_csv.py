import pandas as pd, os, sys

# ----------------------------------------------------------------------
# Paths – adjust only if you move the files
# ----------------------------------------------------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(project_root, "SBE-profiles.csv")   # <-- CSV file
out_path = os.path.join(project_root, "app", "antivenom_lookup.py")

# ----------------------------------------------------------------------
# Read CSV
# ----------------------------------------------------------------------
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded CSV (rows={df.shape[0]}, cols={df.shape[1]})")
except Exception as e:
    sys.exit(f"❌ Failed to read CSV at {csv_path!r}: {e}")

# ----------------------------------------------------------------------
# Detect the species & antivenom columns (case‑insensitive match)
# ----------------------------------------------------------------------
species_col = None
antivenom_col = None
for col in df.columns:
    lowered = str(col).lower()
    if "species" in lowered:
        species_col = col
    if "antivenom" in lowered or "product" in lowered:
        antivenom_col = col

if species_col is None or antivenom_col is None:
    sys.exit(
        f"❌ Could not locate required columns. Found columns: {list(df.columns)}"
    )

# ----------------------------------------------------------------------
# Build the dictionary
# ----------------------------------------------------------------------
mapping = dict(
    zip(
        df[species_col].astype(str).str.strip(),
        df[antivenom_col].astype(str).str.strip(),
    )
)

# ----------------------------------------------------------------------
# Write the Python module
# ----------------------------------------------------------------------
try:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('"""Auto‑generated antivenom lookup dictionary from SBE‑profiles.csv"""\n')
        f.write("ANTIVENOM_MAP = {\n")
        for k, v in mapping.items():
            f.write(f"    {repr(k)}: {repr(v)},\n")
        f.write("}\n")
    print(f"✅ Antivenom lookup written to {out_path}")
except Exception as e:
    sys.exit(f"❌ Failed to write lookup module: {e}")
