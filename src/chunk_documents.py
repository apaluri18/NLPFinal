import pandas as pd
import nltk
import os

# Ensure the punkt tokenizer is available for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def create_dummy_data(path):
    """Creates a sample enterprise CSV if you don't have one ready yet."""
    dummy_data = pd.DataFrame({
        'doc_id': ['KB001', 'KB002'],
        'title': ['User Retention Policy', 'API Rate Limits'],
        'content': [
            "User accounts that have not logged in for 90 days are flagged as inactive. Administrators must manually delete them using the admin console. Automated deletion is currently disabled.",
            "The standard API rate limit is 1000 requests per minute. Exceeding this limit triggers a 429 Too Many Requests error. Enterprise tiers can request a quota increase via the support portal."
        ]
    })
    dummy_data.to_csv(path, index=False)
    print(f"Created sample raw data at {path}")

def chunk_corpus(input_path, output_path):
    print(f"Loading raw documents from {input_path}...")
    
    if not os.path.exists(input_path):
        create_dummy_data(input_path)
        
    df = pd.read_csv(input_path)
    
    if 'content' not in df.columns:
        raise ValueError("The raw CSV must contain a 'content' column.")

    records = []
    for _, row in df.iterrows():
        # Split paragraph content into distinct, atomic statements
        sentences = nltk.sent_tokenize(str(row['content']))
        
        for i, sentence in enumerate(sentences):
            clean_sentence = sentence.strip()
            # Filter out UI artifacts or extremely short string fragments
            if len(clean_sentence) > 15:
                records.append({
                    'doc_id': row.get('doc_id', f'doc_{_}'),
                    'title': row.get('title', 'Untitled'),
                    'chunk_id': f"{row.get('doc_id', f'doc_{_}')}_{i}",
                    'chunk_text': clean_sentence
                })

    processed_df = pd.DataFrame(records)
    processed_df.to_csv(output_path, index=False)
    
    print(f"Processed {len(df)} documents into {len(processed_df)} atomic chunks.")
    print(f"Saved processed corpus to {output_path}")
    print(processed_df.head())

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    RAW_PATH = "data/raw/enterprise_docs.csv"
    PROCESSED_PATH = "data/processed/corpus_chunks.csv"
    
    chunk_corpus(RAW_PATH, PROCESSED_PATH)