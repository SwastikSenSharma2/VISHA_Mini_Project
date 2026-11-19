import pandas as pd, sys
path = r'C:\\Users\\Swastik Sen Sharma\\.gemini\\antigravity\\brain\\0fd3953b-6de3-4d52-833b-ca47239727d1\\SBE-profiles.xlsx'
try:
    df = pd.read_excel(path)
    print('First rows:')
    print(df.head().to_string(index=False))
    print('Columns:', df.columns.tolist())
except Exception as e:
    print('Error reading Excel:', e)
