import argparse
import re
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

NEG_PAT = re.compile(
    r"(does not support|do not support|not sufficient|insufficient|no evidence|"
    r"contradict|cannot be determined|cannot determine|unreliable|not faithful|"
    r"unsupported|does not provide|doesn't support|not directly support|no direct|"
    r"not mentioned|does not mention|no information|fails to support)",
    re.IGNORECASE,
)


def extract_completion(row):
    text = str(row["raw_judge_output"])
    marker = "Reasoning:" if row.get("setting") == "cot" else "Final verdict:"
    return text.split(marker, 1)[1].strip() if marker in text else text.strip()


def parse_strict(comp, setting):
    c = str(comp).lower()
    if setting == "zero_shot":
        m = re.search(r"\b(unfaithful|not supported|faithful|supported)\b", c)
        if not m:
            return 1, True
        return (0 if m.group(1) in ["unfaithful", "not supported"] else 1), False

    fv = re.search(r"final verdict[:\s\*'\"]*\b(unfaithful|faithful)\b", c)
    if fv:
        return (0 if fv.group(1) == "unfaithful" else 1), False

    matches = re.findall(r"\b(unfaithful|not supported|faithful|supported)\b", c)
    if not matches:
        return 1, True
    return (0 if matches[-1] in ["unfaithful", "not supported"] else 1), False


def parse_semantic(comp, setting):
    val, no_match = parse_strict(comp, setting)
    if not no_match:
        return val
    return 0 if NEG_PAT.search(str(comp)) else 1


def main():
    parser = argparse.ArgumentParser(description="Compute agreement and overhead metrics.")
    parser.add_argument(
        "--input_csv",
        default="data/ground_truth/e5_top3_evaluations.csv",
        help="Path to evaluation CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["gt_bin"] = (
        df["ground_truth_label"].astype(str).str.strip().str.lower() == "supported"
    ).astype(int)
    df["completion"] = df.apply(extract_completion, axis=1)

    strict_out = [parse_strict(r["completion"], r["setting"]) for _, r in df.iterrows()]
    df["pred_strict"] = [x[0] for x in strict_out]
    df["no_match"] = [x[1] for x in strict_out]
    df["pred_semantic"] = [
        parse_semantic(r["completion"], r["setting"]) for _, r in df.iterrows()
    ]
    df["truncated_256"] = (df["setting"] == "cot") & (df["completion_tokens"] >= 256)

    summary_rows = []
    for (model_name, setting), grp in df.groupby(["model_name", "setting"], sort=False):
        y_true = grp["gt_bin"].to_numpy()
        y_strict = grp["pred_strict"].to_numpy()
        y_sem = grp["pred_semantic"].to_numpy()

        tn, fp, fn, tp = confusion_matrix(y_true, y_strict, labels=[0, 1]).ravel()
        post_prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        retention = (tp + fp) / len(y_true) * 100.0
        fn_rate = (fn / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        hal_recall = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        summary_rows.append(
            {
                "Model": model_name.split("/")[-1],
                "Setting": setting,
                "Precision": grp["precision"].iloc[0] if "precision" in grp else "FP16",
                "Kappa": round(cohen_kappa_score(y_true, y_strict), 3),
                "Acc": round(accuracy_score(y_true, y_strict), 3),
                "HalRec": round(hal_recall, 3),
                "MacroF1": round(f1_score(y_true, y_strict, average="macro"), 3),
                "PostPrec%": round(post_prec, 1),
                "Ret%": round(retention, 1),
                "FN%": round(fn_rate, 1),
                "CM(TN,FP,FN,TP)": f"({tn},{fp},{fn},{tp})",
                "NoMatch": int(grp["no_match"].sum()),
                "Trunc256": int(grp["truncated_256"].sum()),
                "SemKappa": round(cohen_kappa_score(y_true, y_sem), 3),
                "SemMacroF1": round(f1_score(y_true, y_sem, average="macro"), 3),
                "Lat(s)": round(grp["latency"].mean(), 2),
                "PromptTok": round(grp["prompt_tokens"].mean(), 1),
                "CompTok": round(grp["completion_tokens"].mean(), 1),
            }
        )

    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\n--- 3-Way Ground Truth vs. Strict Prediction ---")
    for (model_name, setting), grp in df.groupby(["model_name", "setting"], sort=False):
        print(f"\n{model_name.split('/')[-1]} [{setting}]")
        print(pd.crosstab(grp["ground_truth_label"], grp["pred_strict"]))


if __name__ == "__main__":
    main()
