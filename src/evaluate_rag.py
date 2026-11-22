import os
import pandas as pd
import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Safely detect if a GPU is available to prevent crashes on CPU-only runtimes
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: Running on CPU. It is highly recommended to switch your Colab runtime to GPU (Runtime -> Change runtime type -> T4 GPU).")

print(f"Loading E5 retriever on {device}...")
retriever = SentenceTransformer('intfloat/e5-small-v2', device=device)

def evaluate_claim(claim, evidence, model, tokenizer):
    prompt = f"Claim: {claim}\nEvidence: {evidence}\nEvaluate faithfulness..."
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=60)
    latency = time.time() - start_time

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result, latency

if __name__ == "__main__":
    # Check if Google Drive is mounted
    if not os.path.exists("/content/drive"):
        print("Google Drive is not mounted. Attempting to mount now...")
        from google.colab import drive
        try:
            drive.mount('/content/drive')
        except Exception as e:
            print(f"Could not mount Google Drive: {e}")

    # Point to the data folders uploaded to the root of your Google Drive
    data_path = "/content/drive/MyDrive/data/processed/corpus_chunks.csv"
    truth_path = "/content/drive/MyDrive/data/raw/enterprise_docs.csv"

    if os.path.exists(data_path) and os.path.exists(truth_path):
        print(f"\nFound data files. Running Colab {device.upper()} batch evaluation with stratified sampling...")

        # 1. Load both the chunks and the raw labels
        df = pd.read_csv(data_path)
        truth_df = pd.read_csv(truth_path)

        # 2. Merge them so we know which claims are hallucinations before sampling
        df = df.merge(truth_df[['title', 'ground_truth_label']], on='title', how='left')

       # 3. Create a binary column to sort by (Exact match to avoid substring trap)
        df['is_faithful'] = df['ground_truth_label'].apply(
            lambda x: 1 if str(x).strip().lower() == "supported" else 0
        )

        # 4. Stratified Sample: Force a 50/50 split (or pull as many negatives as exist)
        positives = df[df['is_faithful'] == 1].sample(min(25, len(df[df['is_faithful'] == 1])), random_state=42)
        negatives = df[df['is_faithful'] == 0].sample(min(25, len(df[df['is_faithful'] == 0])), random_state=42)

        # 5. Combine and shuffle the balanced deck
        df = pd.concat([positives, negatives]).sample(frac=1, random_state=42)
        print(f"Balanced evaluation set created with {len(positives)} faithful and {len(negatives)} unfaithful claims.")

        models_to_test = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct"
        ]

        results = []

        for current_model_id in models_to_test:
            print(f"\n--- Initializing {current_model_id} on {device} ---")
            try:
                tokenizer = AutoTokenizer.from_pretrained(current_model_id)

                # Use appropriate device map based on hardware
                device_map = "auto" if device == "cuda" else "cpu"

                judge_model = AutoModelForCausalLM.from_pretrained(
                    current_model_id,
                    device_map=device_map,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )

                for i, row in df.iterrows():
                    claim = row['title']
                    evidence = row['chunk_text']

                    try:
                        json_result, latency = evaluate_claim(claim, evidence, judge_model, tokenizer)
                        results.append({
                            "doc_id": row['doc_id'],
                            "model_name": current_model_id,
                            "claim": claim,
                            "evidence": evidence,
                            "raw_judge_output": json_result,
                            "latency": latency
                        })
                    except Exception as e:
                        print(f"Failed on index {i} with {current_model_id}: {e}")

                print(f"--- Flushing {current_model_id} memory ---")
                del judge_model
                del tokenizer
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as init_err:
                print(f"Could not load or run model {current_model_id}: {init_err}")

        output_df = pd.DataFrame(results)

        # Save the output directly back to your Google Drive
        output_dir = "/content/drive/MyDrive/data/ground_truth"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/multi_model_evaluations.csv"

        output_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully exported comparisons to {output_file}")
    else:
        print(f"\nData missing. Ensure both corpus_chunks.csv and enterprise_docs.csv are uploaded to data/processed and data/raw in your Drive.")
