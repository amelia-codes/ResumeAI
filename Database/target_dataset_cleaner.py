import pandas as pd

target_dataset = pd.read_excel("Database/target_dataset.xlsx", index_col=0)
target_dataset["Target"] = target_dataset["Target"].astype(str).str.upper() == "TRUE"

print(target_dataset)
target_dataset.to_excel("Database/target_dataset.xlsx")