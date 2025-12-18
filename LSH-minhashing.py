import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import random
import numpy as np
import sys

# Shingling: Convert documents to Boolean matrix

file_path = "similarity.txt"
with open(file_path, "r") as file:
    text = file.read()
    paragraphs = [p for p in text.split('\n') if p.strip() != '']
punctuation = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
filtered_paragraphs = []
for paragraph in paragraphs:
    no_punct = paragraph.translate(punctuation)
    words = no_punct.split()
    filtered = [word for word in words if word.lower() not in stop_words]
    filtered_paragraphs.append(filtered)

def shingle(words, k):
    return set(tuple(words[i:i+k]) for i in range(len(words) - k + 1))
k = 3
paragraph_shingles = [shingle(par, k) for par in filtered_paragraphs]

all_shingles = [shi for para in paragraph_shingles for shi in para]

shingles_set = set(all_shingles)
sorted_shingles = sorted(shingles_set)
shingle_index = {shin: id for id, shin in enumerate(sorted_shingles)}
boolean_matrix = np.zeros((len(sorted_shingles), len(paragraph_shingles)), dtype=int)
for col, p in enumerate(paragraph_shingles):
    for shi in p:
        row = shingle_index[shi]
        boolean_matrix[row, col] = 1


# Min-Hashing: Convert large sets to short signatures, while preserving similarity

number_of_hash_functions = 200
select_rows = 33554
rows, cols = boolean_matrix.shape
signature_matrix = np.zeros((number_of_hash_functions,cols),dtype=int)

# Permutations for hash functions
for i in range(number_of_hash_functions):
    permuted_rows = np.random.permutation(rows)
    for col in range(cols):
        min_row = 0
        for r in permuted_rows:
            if boolean_matrix[r,col]:
                min_row = r
                break
        signature_matrix[i,col] = min_row
      
signature_matrix = signature_matrix.astype(int)


# Locality-Sensitive Hashing: Focus on pairs of signatures likely to be from similar documents 

b,ro = 20,10
threshold = (1/b)**(1/ro)
candidates = set()
seen_buckets = [{} for _ in range(b)] 

for col in range(signature_matrix.shape[1]):  
    for band in range(b):
        start = band * r
        end = (band + 1) * r
        band_tuple = tuple(signature_matrix[start:end, col])

        bucket = hash(band_tuple)
        band_dict = seen_buckets[band]
        if bucket in band_dict:
            for other_col in band_dict[bucket]:
                candidates.add((min(col, other_col), max(col, other_col)))
            band_dict[bucket].append(col)
        else:
            band_dict[bucket] = [col]

for i, j in candidates:
    matches = np.sum(signature_matrix[:, i] == signature_matrix[:, j])
    jaccard = matches / signature_matrix.shape[0]
jaccard_values = []

for i, j in candidates:
    matches = np.sum(signature_matrix[:, i] == signature_matrix[:, j])
    jaccard = matches / signature_matrix.shape[0]

if jaccard_values:
    avg_jaccard = sum(jaccard_values) / len(jaccard_values)
else:
    avg_jaccard = 0

print(f"Jaccard similarity of similar pairs: {avg_jaccard}")








            
