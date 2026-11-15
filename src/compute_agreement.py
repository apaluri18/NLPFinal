import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score

# 1. Load the evaluations from your local directory
file_path = "data/ground_truth/multi_model_evaluations.csv"
df = pd.read_csv(file_path)

# 2. Extract the binary prediction from the LLM's raw output
def extract_prediction(text):
    text = str(text).lower()
    if "unfaithful" in text or "not supported" in text or "false" in text:
        return 0
    return 1

df['llm_prediction'] = df['raw_judge_output'].apply(extract_prediction)

# 3. Ensure human ground truth is available
if 'ground_truth' not in df.columns:
    print("⚠️ 'ground_truth' column missing! Using mock data for demonstration.")
    df['ground_truth'] = np.random.choice([0, 1], size=len(df))

# 4. Calculate metrics per model
results = []
for model in df['model_name'].unique():
    model_df = df[df['model_name'] == model]

    kappa = cohen_kappa_score(model_df['ground_truth'], model_df['llm_prediction'])
    acc = accuracy_score(model_df['ground_truth'], model_df['llm_prediction'])
    avg_latency = model_df['latency'].mean()

    results.append({
        "Model": model.split('/')[-1],
        "Cohen's Kappa": round(kappa, 3),
        "Accuracy": round(acc, 3),
        "Avg Latency (s)": round(avg_latency, 2)
    })

# 5. Display the final NeurIPS-ready table
metrics_df = pd.DataFrame(results).sort_values(by="Cohen's Kappa", ascending=False)
print("\n### Multi-Model RAG Evaluation Results")
print(metrics_df.to_markdown(index=False))
