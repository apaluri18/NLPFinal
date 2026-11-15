import os
import pandas as pd
import torch
import time
import json
import gc # Add this for memory management
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# 1. Initialize E5 Retriever
print("Loading E5 retriever on MPS...")
retriever = SentenceTransformer('intfloat/e5-small-v2', device='mps')

def evaluate_claim(claim, evidence, model, tokenizer):
    prompt = f"Claim: {claim}\nEvidence: {evidence}\nEvaluate faithfulness..."
    inputs = tokenizer([prompt], return_tensors="pt").to("mps")

    start_time = time.time()
    outputs = model.generate(**inputs, max_new_tokens=60)
    latency = time.time() - start_time

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result, latency

if __name__ == "__main__":
    data_path = "data/processed/corpus_chunks.csv"

    if os.path.exists(data_path):
        print(f"\nFound {data_path}. Running Apple Silicon batch evaluation...")
        df = pd.read_csv(data_path).sample(50, random_state=42)

        models_to_test = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct"
        ]

        results = []

        for current_model_id in models_to_test:
            print(f"\n--- Initializing {current_model_id} on MPS ---")
            tokenizer = AutoTokenizer.from_pretrained(current_model_id)

            judge_model = AutoModelForCausalLM.from_pretrained(
                current_model_id,
                device_map="mps",
                torch_dtype=torch.float16
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

            # --- CRITICAL MEMORY FLUSH ---
            # Delete the model and tokenizer objects from Python's memory
            del judge_model
            del tokenizer

            # Force garbage collection
            gc.collect()

            # Empty the Mac's Metal Performance Shaders (MPS) cache
            torch.mps.empty_cache()
            print(f"--- Flushed {current_model_id} from memory ---")

        output_df = pd.DataFrame(results)
        os.makedirs("data/ground_truth", exist_ok=True)
        output_df.to_csv("data/ground_truth/multi_model_evaluations.csv", index=False)
        print("\nSuccessfully exported hardware-accelerated comparisons to data/ground_truth/multi_model_evaluations.csv")
    else:
        print(f"\nNo data found at {data_path}. Run chunk_documents.py first.")
