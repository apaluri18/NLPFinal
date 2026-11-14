import pandas as pd
import numpy as np
from datasets import load_dataset
import os

def download_wice_to_csv():
    print("Downloading WICE dataset from Hugging Face...")
    
    # WICE is hosted on several mirrors; tasksource/wice is a reliable compiled version
    dataset = load_dataset("tasksource/wice", split="train")
    df = dataset.to_pandas()
    
    print(f"Downloaded {len(df)} records. Formatting for chunker...") # type: ignore

    # The WICE dataset stores 'evidence' as a list of strings. 
    # We need to join them into a single text block for the chunker.
    def parse_evidence(evidence):
        if isinstance(evidence, (list, np.ndarray)):
            return ' '.join(evidence)
        return str(evidence)

    # Map the WICE schema to our expected RAG schema
    formatted_df = pd.DataFrame({
        'doc_id': [f"wice_{i}" for i in range(len(df))], # type: ignore
        'title': df['claim'], # We'll store the original claim as the 'title' # type: ignore
        'content': df['evidence'].apply(parse_evidence), # type: ignore
        'ground_truth_label': df['label'] # useful later for validation# type: ignore
    })
    
    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/enterprise_docs.csv"
    
    formatted_df.to_csv(output_path, index=False)
    print(f"Successfully saved formatted WICE dataset to {output_path}")
    print(formatted_df.head(2))

if __name__ == "__main__":
    download_wice_to_csv()