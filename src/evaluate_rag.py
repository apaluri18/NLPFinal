import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from sentence_transformers import SentenceTransformer
import time
import os

# 1. Initialize E5 Retriever
print("Loading E5 retriever...")
retriever = SentenceTransformer('intfloat/e5-small-v2')

# 2. Initialize Qwen Judge 
print("Loading Qwen Judge...")
model_id = "Qwen/Qwen2.5-0.5B-Instruct" # Or try "Qwen/Qwen2.5-0.8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
judge_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16
)

# 3. Define the System Prompt
SYS_PROMPT = """You are an expert evaluator assessing the factuality of an AI assistant's response against official enterprise documentation. 
Compare the Atomic Claim against the Retrieved Evidence and output ONLY a JSON object with 'relevance_score' (1-5), 'faithfulness_label' (Entailed, Contradicted, Neutral), and 'rationale'.

Atomic Claim: {claim}
Retrieved Evidence: {evidence}
"""

def evaluate_claim(claim, evidence):
    prompt = SYS_PROMPT.format(claim=claim, evidence=evidence)
    messages = [
        {"role": "system", "content": "You output strict JSON."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(judge_model.device)
    
    start_time = time.time()
    outputs = judge_model.generate(**inputs, max_new_tokens=150, temperature=0.1) # type: ignore
    latency = time.time() - start_time
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response, latency



# Test the pipeline
if __name__ == "__main__":
    test_claim = "The system automatically deletes inactive users after 90 days."
    test_evidence = "User accounts that have not logged in for 90 days are flagged as inactive. Administrators must manually delete them."
    
    print("Testing judge execution...")
    result, latency = evaluate_claim(test_claim, test_evidence)
    print(f"\nExecution Time: {latency:.2f}s")
    print(f"Result:\n{result}")

    # --- NEW BATCH EXECUTION LOGIC ---
    data_path = "data/processed/corpus_chunks.csv"
    if os.path.exists(data_path):
        print(f"\nFound {data_path}. Running batch evaluation on 50 samples...")
        df = pd.read_csv(data_path).sample(50, random_state=42)
        
        results = []
        for i, row in df.iterrows():
            claim = row['title'] # In WICE, the 'title' column holds the claim
            evidence = row['chunk_text']
            
            # (In a full pipeline, you'd run E5 here to retrieve the evidence)
            # For this baseline test, WICE already pairs the claim with the evidence
            
            try:
                json_result, latency = evaluate_claim(claim, evidence)
                results.append({
                    "doc_id": row['doc_id'],
                    "claim": claim,
                    "evidence": evidence,
                    "raw_judge_output": json_result,
                    "latency": latency
                })
            except Exception as e:
                print(f"Failed on index {i}: {e}")
                
        output_df = pd.DataFrame(results)
        os.makedirs("data/ground_truth", exist_ok=True)
        output_df.to_csv("data/ground_truth/judge_evaluations.csv", index=False)
        print("Successfully exported 50 evaluations to data/ground_truth/judge_evaluations.csv")
    else:
        print(f"\nNo data found at {data_path}. Run chunk_documents.py first.")