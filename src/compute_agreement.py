import re
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, f1_score

# Switch between "multi_model_evaluations_3.csv" (Zero-Shot) and "multi_model_evaluations_cot_4.csv" (CoT)
file_path = sys.argv[1] if len(sys.argv) > 1 else "data/ground_truth/multi_model_evaluations.csv"
df = pd.read_csv(file_path)

def extract_prediction(text):
    text = str(text)
    # Strip the echoed prompt if this is a CoT output
    if "Reasoning:" in text:
        generated = text.split("Reasoning:", 1)[-1].lower()
    elif "Evaluate faithfulness..." in text:
        generated = text.split("Evaluate faithfulness...", 1)[-1].lower()
    else:
        generated = text.lower()

    # Find the LAST explicit verdict mentioned in the generated response
    matches = re.findall(r"\b(unfaithful|not supported|faithful|supported)\b", generated)
    if matches:
        last_verdict = matches[-1]
        return 0 if last_verdict in ("unfaithful", "not supported") else 1
    return 1  # Default fallback if model fails to output a clear verdict

df["llm_prediction"] = df["raw_judge_output"].apply(extract_prediction)

# Merge human ground truth with strict binary matching
truth_df = pd.read_csv("data/raw/enterprise_docs.csv")
df = df.merge(truth_df[["title", "ground_truth_label"]], left_on="claim", right_on="title", how="left")
df["ground_truth"] = df["ground_truth_label"].apply(
    lambda x: 1 if str(x).strip().lower() == "supported" else 0
)

print("\n--- Distribution Check ---")
print("Ground Truth:\n", df["ground_truth"].value_counts().to_dict())
print("LLM Predictions:\n", df["llm_prediction"].value_counts().to_dict())

results = []
for model in df["model_name"].unique():
    m_df = df[df["model_name"] == model]
    y_true = m_df["ground_truth"]
    y_pred = m_df["llm_prediction"]

    results.append({
        "Model": model.split("/")[-1],
        "Cohen's Kappa": round(cohen_kappa_score(y_true, y_pred), 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Hallucination Recall": round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 3),
        "Macro-F1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 3),
        "Avg Latency (s)": round(m_df["latency"].mean(), 2),
    })

metrics_df = pd.DataFrame(results).sort_values(by="Cohen's Kappa", ascending=False)
print(f"\n### Results for {file_path}")
print(metrics_df.to_markdown(index=False))
