from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json
import faiss

app = Flask(__name__)
CORS(app)

# Load MedCPT Query Encoder
query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

# Load Faiss index and patient data
index = faiss.read_index('patient_embeddings.faiss')
with open('processed_patients.json', 'r') as f:
    patients = json.load(f)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data['query']
    page = data.get('page', 1)
    per_page = 10

    # Encode the query
    inputs = query_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = query_model(**inputs).last_hidden_state[:, 0, :].numpy()
    
    # Perform the search
    k = page * per_page
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    # Prepare the results
    start = (page - 1) * per_page
    end = start + per_page
    page_indices = indices[0][start:end]
    page_distances = distances[0][start:end]
    
    results = [
        {
            "summary": patients[int(i)]['patient'],
            "score": float(1 / (1 + d)),
            "age": patients[int(i)].get('age', 'N/A'),
            "gender": patients[int(i)].get('gender', 'N/A')
        }
        for i, d in zip(page_indices, page_distances)
    ]
    
    return jsonify({
        "results": results,
        "has_more": len(indices[0]) > end
    })

if __name__ == '__main__':
    app.run(debug=True)
