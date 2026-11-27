import gc
import os
import time
import numpy as np
import pandas as pd
import torch
from google.colab import userdata
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def chunk_text(text, chunk_words=80, stride=40):
    """Split text into overlapping word windows while skipping short fragments."""
    words = str(text).split()
    if len(words) <= chunk_words:
        return [" ".join(words)] if len(str(text).strip()) > 30 else []
    chunks = []
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start : start + chunk_words]).strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        if start + chunk_words >= len(words):
            break
    return chunks


def build_prompt(claim, evidence, mode):
    if mode == "zero_shot":
        return (
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n"
            "Task: Decide if the claim is strictly supported by the evidence.\n"
            "Answer with exactly one word: Faithful or Unfaithful.\n"
            "Final verdict:"
        )
    return (
        f"Claim: {claim}\n"
        f"Evidence: {evidence}\n"
        "Task: Evaluate if the claim is strictly supported by the evidence.\n"
        "First, think step-by-step and analyze the evidence briefly. "
        "Then end with 'Final verdict: Faithful' or 'Final verdict: Unfaithful'.\n"
        "Reasoning:"
    )


def run_judge(claim, evidence, mode, model, tokenizer, device):
    prompt = build_prompt(claim, evidence, mode)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    prompt_tokens = int(inputs["input_ids"].shape[1])
    max_new = 16 if mode == "zero_shot" else 256

    start_t = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start_t
    completion_tokens = int(outputs.shape[1] - prompt_tokens)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded, latency, prompt_tokens, completion_tokens


def main():
    assert torch.cuda.is_available(), "CUDA GPU is required to run this evaluation script."
    device = "cuda"

    if not os.path.exists("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            hf_token = userdata.get("HF_TOKEN")
            os.environ["HF_TOKEN"] = hf_token
        except Exception:
            hf_token = None
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)

    raw_docs_path = os.environ.get("RAW_DOCS_PATH", "/content/drive/MyDrive/data/raw/enterprise_docs.csv")
    prev_oracle_path = os.environ.get(
        "PREV_ORACLE_PATH", "/content/drive/MyDrive/data/ground_truth/oracle_7b_cot.csv"
    )
    output_file = os.environ.get(
        "OUTPUT_FILE", "/content/drive/MyDrive/data/ground_truth/e5_top3_evaluations.csv"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    docs_df = pd.read_csv(raw_docs_path)
    if os.path.exists(prev_oracle_path):
        target_ids = pd.read_csv(prev_oracle_path)["doc_id"].unique().tolist()
        eval_df = docs_df[docs_df["doc_id"].isin(target_ids)].copy()
    else:
        docs_df["is_faithful"] = (
            docs_df["ground_truth_label"].astype(str).str.strip().str.lower() == "supported"
        ).astype(int)
        pos = docs_df[docs_df["is_faithful"] == 1].sample(25, random_state=42)
        neg = docs_df[docs_df["is_faithful"] == 0].sample(25, random_state=42)
        eval_df = pd.concat([pos, neg]).sample(frac=1, random_state=42).copy()

    retriever = SentenceTransformer("intfloat/e5-small-v2", device=device)
    top_k = 3
    retrieved_texts, top1_scores = [], []

    for _, row in eval_df.iterrows():
        claim = str(row["title"]).strip()
        chunks = chunk_text(row["content"], chunk_words=80, stride=40)
        q_emb = retriever.encode([f"query: {claim}"], normalize_embeddings=True, show_progress_bar=False)
        p_embs = retriever.encode(
            [f"passage: {c}" for c in chunks],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sims = (p_embs @ q_emb[0]).astype(float)
        top_idx = np.argsort(sims)[::-1][:top_k]
        retrieved_texts.append(" [...] ".join([chunks[idx] for idx in top_idx]))
        top1_scores.append(float(sims[top_idx[0]]))

    eval_df["retrieved_evidence"] = retrieved_texts
    eval_df["top1_cosine_sim"] = top1_scores

    del retriever
    gc.collect()
    torch.cuda.empty_cache()

    model_configs = [
        {"id": "Qwen/Qwen2.5-0.5B-Instruct", "modes": ["zero_shot", "cot"], "quant4bit": False},
        {"id": "Qwen/Qwen2.5-1.5B-Instruct", "modes": ["zero_shot", "cot"], "quant4bit": False},
        {"id": "meta-llama/Llama-3.2-1B-Instruct", "modes": ["zero_shot", "cot"], "quant4bit": False},
        {"id": "Qwen/Qwen2.5-7B-Instruct", "modes": ["cot"], "quant4bit": True},
    ]

    results = []
    for cfg in model_configs:
        model_id = cfg["id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if cfg["quant4bit"]:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            judge_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", quantization_config=bnb_cfg
            )
        else:
            judge_model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16
            )
        judge_model.eval()

        for mode in cfg["modes"]:
            for _, row in eval_df.iterrows():
                raw_out, lat, p_tok, c_tok = run_judge(
                    row["title"], row["retrieved_evidence"], mode, judge_model, tokenizer, device
                )
                results.append(
                    {
                        "doc_id": row["doc_id"],
                        "ground_truth_label": row["ground_truth_label"],
                        "model_name": model_id,
                        "setting": mode,
                        "precision": "4-bit NF4" if cfg["quant4bit"] else "FP16",
                        "top1_cosine_sim": row["top1_cosine_sim"],
                        "claim": row["title"],
                        "evidence": row["retrieved_evidence"],
                        "raw_judge_output": raw_out,
                        "prompt_tokens": p_tok,
                        "completion_tokens": c_tok,
                        "latency": lat,
                    }
                )
            pd.DataFrame(results).to_csv(output_file, index=False)

        del judge_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Saved {len(results)} evaluation rows to {output_file}")


if __name__ == "__main__":
    main()
