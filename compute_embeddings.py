import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss
import sys

# Load MedCPT Article Encoder
article_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
article_model.to(device)

def load_pmc_patients(file_path, max_samples=None):
    df = pd.read_csv(file_path)
    if max_samples:
        df = df.head(max_samples)
    return df

def compute_patient_embeddings(patients, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(patients), batch_size), desc="Computing embeddings"):
        batch = patients[i:i + batch_size]
        inputs = article_tokenizer(batch['patient'].tolist(), return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        
        with torch.no_grad():
            batch_embeddings = article_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    return np.vstack(embeddings)

print("Loading PMC-Patients dataset...")
patients = load_pmc_patients('pmc_patients.csv', max_samples=160000)  # Adjust max_samples as needed

print("Computing patient embeddings in batches...")
sys.stdout.flush()
patient_embeddings = compute_patient_embeddings(patients)

print("Creating Faiss index...")
dimension = patient_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(patient_embeddings.astype('float32'))

print("Saving Faiss index...")
faiss.write_index(index, 'patient_embeddings.faiss')

print("Saving processed patients data...")
patients.to_json('processed_patients.json', orient='records')

print("Embeddings computed and indexed.")
