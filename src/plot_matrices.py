import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# Paths
ZS_PATH = "data/ground_truth/multi_model_evaluations.csv"
COT_PATH = "data/ground_truth/multi_model_evaluations_cot.csv"
ORACLE_PATH = "data/ground_truth/oracle_7b_cot.csv"
TRUTH_PATH = "data/raw/enterprise_docs.csv"
OUTPUT_PATH = "confusion_matrices_comparison.png"


def extract_prediction(text):
    text = str(text)
    if "Reasoning:" in text:
        generated = text.split("Reasoning:", 1)[-1].lower()
    elif "Evaluate faithfulness..." in text:
        generated = text.split("Evaluate faithfulness...", 1)[-1].lower()
    else:
        generated = text.lower()

    matches = re.findall(r"\b(unfaithful|not supported|faithful|supported)\b", generated)
    if matches:
        last_verdict = matches[-1]
        return 0 if last_verdict in ("unfaithful", "not supported") else 1
    return 1


def load_and_prepare(eval_path, truth_df):
    df = pd.read_csv(eval_path)
    df["llm_prediction"] = df["raw_judge_output"].apply(extract_prediction)
    df = df.merge(
        truth_df[["title", "ground_truth_label"]],
        left_on="claim",
        right_on="title",
        how="left",
    )
    df["ground_truth"] = df["ground_truth_label"].apply(
        lambda x: 1 if str(x).strip().lower() == "supported" else 0
    )
    return df


def plot_cm(ax, y_true, y_pred, title, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    kappa = cohen_kappa_score(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=False,
        square=True,
        xticklabels=["Unfaithful", "Faithful"],
        yticklabels=["Unfaithful", "Faithful"],
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"{title}\n($\\kappa = {kappa:.2f}$)", fontsize=11, pad=8)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Ground Truth", fontsize=10)


def main():
    truth_df = pd.read_csv(TRUTH_PATH)
    zs_df = load_and_prepare(ZS_PATH, truth_df)
    cot_df = load_and_prepare(COT_PATH, truth_df)
    oracle_df = load_and_prepare(ORACLE_PATH, truth_df)

    sub_2b_models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B"),
        ("meta-llama/Llama-3.2-1B-Instruct", "Llama-3.2-1B"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.suptitle(
        "RAG Faithfulness Confusion Matrices: Zero-Shot vs. Chain-of-Thought (Balanced WICE, N=50)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Row 0: Zero-Shot (Sub-2B)
    for col, (model_id, short_name) in enumerate(sub_2b_models):
        m_df = zs_df[zs_df["model_name"] == model_id]
        plot_cm(
            axes[0, col],
            m_df["ground_truth"],
            m_df["llm_prediction"],
            f"{short_name} (Zero-Shot)",
            cmap="Reds",
        )
    axes[0, 3].axis("off")

    # Row 1: Chain-of-Thought (Sub-2B + 7B Oracle)
    for col, (model_id, short_name) in enumerate(sub_2b_models):
        m_df = cot_df[cot_df["model_name"] == model_id]
        plot_cm(
            axes[1, col],
            m_df["ground_truth"],
            m_df["llm_prediction"],
            f"{short_name} (CoT)",
            cmap="Blues",
        )

    plot_cm(
        axes[1, 3],
        oracle_df["ground_truth"],
        oracle_df["llm_prediction"],
        "Qwen2.5-7B 4-bit (CoT Oracle)",
        cmap="Greens",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Saved confusion matrix figure to {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
