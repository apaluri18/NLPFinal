import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from sentence_transformers import SentenceTransformer
import time

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