import os
import sqlite3
import pandas as pd

# ---------------- FILE PATHS ---------------- #

CSV_FILE = "cleaned_schemes.csv"
DB_FILE = "schemes.db"

# ---------------- DELETE OLD DATABASE ---------------- #

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)
    print("Old database deleted.")

# ---------------- READ CSV ---------------- #

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df)} records from CSV.")

# ---------------- CREATE DATABASE ---------------- #

conn = sqlite3.connect(DB_FILE)

# Create table automatically
df.to_sql(
    "schemes",
    conn,
    if_exists="replace",
    index=False
)

conn.commit()

# ---------------- VERIFY DATABASE ---------------- #

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM schemes")
count = cursor.fetchone()[0]

print(f"Database created successfully!")
print(f"Rows inserted: {count}")

print("\nColumns:")
for col in df.columns:
    print(f"- {col}")

conn.close()

print("\nDatabase saved as schemes.db")