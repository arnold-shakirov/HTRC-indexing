import os
import json
import bz2
import random
import numpy as np
from collections import Counter
from sklearn.decomposition import IncrementalPCA
import scipy as sp
import faiss
from rich.progress import Progress
from rich.console import Console

# Constants
DATA_DIR = '/data/htrc'
SAMPLE_SIZE = 2000
VEC_SAMPLE_SIZE = 40000
WORD_LIST_SIZE = 10000
PCA_COMPONENTS = 128
FAISS_CLUSTERS = 1024
FAISS_BITS = 16

console = Console()

# Function to load a random sample of books
def load_random_sample(sample_size):
    global all_files
    all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(DATA_DIR) for f in filenames if f.endswith('.json.bz2')]
    random.shuffle(all_files)
    sample_files = all_files[:sample_size]
    return sample_files

# Function to create a word frequency vector for a book
def create_word_vector(book_data, common_words):
    word_counts = Counter()
    for page in book_data.get('features', {}).get('pages', []):
        if 'body' in page and page['body'] is not None and 'tokenPosCount' in page['body'] and page['body']['tokenPosCount'] is not None:
            for word, count in page['body']['tokenPosCount'].items():
                #if word in common_words:
                word_counts[word] += sum(count.values())
    total_words = sum(word_counts.values())
    vector = [word_counts[word] / total_words if word in word_counts else 0 for word in common_words]
    return vector

# Load a random sample of books and determine the most common words
with Progress() as progress:
    task = progress.add_task("[green]Loading sample files...", total=SAMPLE_SIZE)
    sample_files = load_random_sample(SAMPLE_SIZE)
    all_words = Counter()
    for file in sample_files:
        with bz2.open(file, 'rt') as f:
            book_data = json.load(f)
            for page in book_data.get('features', {}).get('pages', []):
                if 'body' in page and page['body'] is not None and 'tokenPosCount' in page['body'] and page['body']['tokenPosCount'] is not None:
                    for word, count in page['body']['tokenPosCount'].items():
                        all_words[word] += sum(count.values())
        progress.update(task, advance=1)

common_words = [word for word, count in all_words.most_common(WORD_LIST_SIZE)]
with open('common_words.txt', 'w') as f:
    f.write('\n'.join(common_words))

# Train a PCA model on the random subset vectors
vectors = []
vec_sample_files = load_random_sample(VEC_SAMPLE_SIZE)
with Progress() as progress:
    task = progress.add_task("[green]Creating word vectors...", total=VEC_SAMPLE_SIZE)
    for file in vec_sample_files:
        with bz2.open(file, 'rt') as f:
            book_data = json.load(f)
            vector = create_word_vector(book_data, common_words)
            vectors.append(vector)
        progress.update(task, advance=1)

# use scipy sparse matrix
sparse_vectors = sp.sparse.csr_matrix(vectors)

console.print('Training PCA model...')
pca = IncrementalPCA(n_components=PCA_COMPONENTS)
pca_vectors = pca.fit_transform(sparse_vectors)
with open('pca_model.npy', 'wb') as f:
    np.save(f, pca.components_)

# Initialize a Faiss index
faiss_index = faiss.index_factory(PCA_COMPONENTS, f'IVF{FAISS_CLUSTERS},PQ{FAISS_BITS}')
faiss_index.train(np.array(pca_vectors).astype(np.float32))

# Process each book and add it to the Faiss index
book_ids = []
with Progress() as progress:
    task = progress.add_task("[green]Processing books for Faiss index...", total=len(all_files))
    for i, file in enumerate(all_files):
        with bz2.open(file, 'rt') as f:
            book_data = json.load(f)
            vector = create_word_vector(book_data, common_words)
            pca_vector = pca.transform([vector])[0]
            faiss_index.add(np.array([pca_vector]).astype(np.float32))
        book_ids.append(file)
        progress.update(task, advance=1)
        if i % 1000 == 0 or i == len(all_files) - 1:
            console.log(f'Processed {i + 1} books')
            faiss.write_index(faiss_index, 'faiss_index.bin')
            with open('book_ids.txt', 'w') as f:
                f.write('\n'.join(book_ids))

# Function to query similar books
def query_similar_books(query_file, k=10):
    with bz2.open(query_file, 'rt') as f:
        book_data = json.load(f)
        vector = create_word_vector(book_data, common_words)
        pca_vector = pca.transform([vector])[0]
        distances, indices = faiss_index.search(np.array([pca_vector]).astype(np.float32), k)
        similar_books = [book_ids[idx] for idx in indices[0]]
        return similar_books

# Example query
query_file = random.choice(all_files)
similar_books = query_similar_books(query_file, k=5)
console.log(f'Similar books to {query_file}:')
for book in similar_books:
    console.log(book)

