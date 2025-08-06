import sqlite3
from pathlib import Path

# Create SQLite DB
conn = sqlite3.connect("ONET_DATABASE.db")
cursor = conn.cursor()

# Path to folder with your .sql files
sql_folder = Path().cwd()

# Execute each SQL file
for sql_file in sql_folder.glob("*.sql"):
    print("Building table from file:", sql_file)
    with open(sql_file, "r") as f:
        sql_script = f.read()
        cursor.executescript(sql_script)  # Run DDL/DML in file
    print("Built table from file:", sql_file)

conn.commit()
conn.close()


conn = sqlite3.connect("ONET_DATABASE.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", tables)

for table_name, in tables:
    print(f"Data in {table_name}:")
    rows = cursor.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()
    for row in rows:
        print(row)