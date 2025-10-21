import pandas as pd
import os

# Папка, где лежат ваши результаты
RESULTS_DIR = "/content/results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "combined_history.csv")

# Все активации, которые вы обучали
activations = ["sigmoid", "tanh", "relu"]

combined = []

for act in activations:
    file_path = os.path.join(RESULTS_DIR, f"{act}_history.csv")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data["activation"] = act.upper()
        combined.append(data)
    else:
        print(f"⚠️ File not found: {file_path}")

# Объединяем все в один DataFrame
if combined:
    full_df = pd.concat(combined, ignore_index=True)
    full_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Combined CSV saved: {OUTPUT_FILE}")
    print(full_df.head())
else:
    print("❌ No CSV files found to combine.")
