import pandas as pd
# import matplotlib.pyplot as plt

# target_dataset = pd.read_excel("Database/target_dataset.xlsx", index_col=0)
# target_dataset["Target"] = target_dataset["Target"].astype(str).str.upper() == "TRUE"

# target_insights = target_dataset.groupby("Phrase")["Target"].agg(["max", "count"]).sort_values("count", ascending=False)
# print(target_insights[(target_insights["count"] > 1) & (target_insights["max"] == True)].head(10))

# plt.hist(target_insights[(target_insights["count"] > 1) & (target_insights["max"] == True)]["count"])
# plt.title("phrase count pmf (excluding the 350 that occur once)")
# plt.xlabel("Phrase Count")
# plt.ylabel("PROBABILITY")
# plt.show()

# # print(target_dataset)
# # target_dataset.to_excel("Database/target_dataset.xlsx")

# target_insights.loc["Knowledge", "max"] = False

# print(target_insights)
# target_insights.rename(columns={"max": "target"})["target"].to_csv("Database/target_dataset_2.csv")
df = pd.concat([pd.read_csv("Database/target_dataset_2.csv"),pd.read_csv("Database/target_addition_false.csv", encoding_errors="replace").rename(columns={"word":"Phrase", "false/true":"target"})], axis=0, ignore_index=True)
df.to_csv("Database/combined_targets.csv")