
import os
import gdown
import time
import pandas as pd
from .model_loading import load_model
import numpy as np
from tqdm import tqdm
import gc


def download_data_from_gdrive(url,output_path):
    if os.path.exists(output_path):
        start_time = time.time()
        # Start downloading
        print(f"Downloading file from: {url}")
        gdown.download(url,output_path,quiet=False,fuzzy=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Download completed in {elapsed_time:.2f} seconds.")
            
    else:
        os.makedirs(output_path)
        start_time = time.time()
        # Start downloading
        print(f"Downloading file from: {url}")
        gdown.download(url,output_path,quiet=False,fuzzy=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Download completed in {elapsed_time:.2f} seconds.")

def read_part_file(file_path,chunksize,all_embeddings=[],all_ratings=[]):
    column_names= ['rating', 'sum','review']
    chunk_iter = pd.read_csv(file_path, chunksize=chunksize,names=column_names, header=None)
    embedding_type_model="UAE" # voyageai
    temp_dir = 'temp_chunks'
    model=load_model(embedding_type_model)
    os.makedirs(temp_dir, exist_ok=True)
    chunk_idx = 0
    for chunk in tqdm(chunk_iter):
        chunk = chunk.drop(columns=['sum'])
        ratings=chunk["rating"].to_numpy()
        reviews=chunk["review"].to_list()
        if embedding_type_model== "UAE":
            doc_vecs = model.encode(reviews, normalize_embedding=True)
        if embedding_type_model== "voyageai":
            doc_vecs = model.embed(reviews,model="voyage-large-2-instruct",input_type="document")
        chunk_data = np.hstack((ratings.reshape(-1, 1), doc_vecs))
        np.save(os.path.join(temp_dir, f'chunk_{chunk_idx}.npy'), chunk_data)
        chunk_idx += 1
        del ratings, reviews, chunk, doc_vecs, chunk_data
        gc.collect()
    # Combine all saved chunks into one file
    combined_array = []
    for i in range(chunk_idx):
        chunk_data = np.load(os.path.join(temp_dir, f'chunk_{i}.npy'))
        combined_array.append(chunk_data)
        os.remove(os.path.join(temp_dir, f'chunk_{i}.npy'))
        del i , chunk_data
        gc.collect()

    combined_array = np.vstack(combined_array)
    np.save('ratings_embeddings.npy', combined_array)
  




       

