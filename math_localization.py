import csv
import math
import os
import matplotlib.pyplot as plt
from statistics import median
from scipy.ndimage import gaussian_filter1d
import numpy as np

# === CONFIG ===
SIGNATURE_FOLDER = "signatures/"
TEST_SIGNATURE_PATH = "test_signature.csv"

# Define neighbor map
neighbors = {
    'signature1': ['signature2', 'signature4'],
    'signature2': ['signature1', 'signature3', 'signature5'],
    'signature3': ['signature2', 'signature6'],
    'signature4': ['signature1', 'signature5'],
    'signature5': ['signature4', 'signature6', 'signature2'],
    'signature6': ['signature5', 'signature3'],
    'signature7': ['signature5', 'signature8'],
    'signature8': ['signature7', 'signature9'],
    'signature9': ['signature8']
}

# === FUNCTIONS ===

def load_signature(filepath):
    with open(filepath, newline='') as file:
        reader = csv.reader(file)
        next(reader)
        raw = [int(row[1]) for row in reader]
        filtered = gaussian_filter1d(raw, sigma=1.0).tolist()
        return filtered

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (norm_a * norm_b)

def best_cosine_match(test_sig, known_sigs):
    all_cosine = {}
    all_euclidean = {}
    length = len(test_sig)

    for name, sig in known_sigs.items():
        best_sim = -1
        best_shift = 0
        for shift in range(length):
            shifted = test_sig[shift:] + test_sig[:shift]
            sim = cosine_similarity(shifted, sig)
            if sim > best_sim:
                best_sim = sim
                best_shift = shift
        all_cosine[name] = best_sim
        shifted_test = test_sig[best_shift:] + test_sig[:best_shift]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(shifted_test, sig)))
        all_euclidean[name] = dist
    return all_cosine, all_euclidean

# === MAIN ===

# === Load test signature ===
test_signature = load_signature(TEST_SIGNATURE_PATH)

# === Load all known signatures ===
known_signatures = {}
for filename in sorted(os.listdir(SIGNATURE_FOLDER)):
    if filename.endswith(".csv") and filename != "test_signature.csv":
        path = os.path.join(SIGNATURE_FOLDER, filename)
        known_signatures[filename.replace('.csv', '')] = load_signature(path)

# === Get matching scores ===
all_cosine, all_euclidean = best_cosine_match(test_signature, known_signatures)

# === Combine Cosine and Euclidean scores ===
min_euc = min(all_euclidean.values())
max_euc = max(all_euclidean.values())

combined_scores = {}
for name in all_cosine.keys():
    cos = all_cosine[name]
    euc = all_euclidean[name]
    norm_euc = (max_euc - euc) / (max_euc - min_euc)
    combined_score = cos + norm_euc
    combined_scores[name] = combined_score

# === Normalize combined scores ===
max_combined = max(combined_scores.values())
normalized_scores = {k: v / max_combined for k, v in combined_scores.items()}

# === Find best match ===
sorted_normalized = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
best_name, best_norm = sorted_normalized[0]

# === Find best neighbor among real neighbors ===
neighbor_list = neighbors.get(best_name, [])
neighbor_scores = {n: normalized_scores.get(n, 0) for n in neighbor_list if n in normalized_scores}

threshold_close = 0.90  # 90% similarity required

if neighbor_scores:
    second_best_neighbor = max(neighbor_scores, key=neighbor_scores.get)
    second_best_score = neighbor_scores[second_best_neighbor]

    total = best_norm + second_best_score
    rel_position = second_best_score / total

    similarity_ratio = second_best_score / best_norm

    if similarity_ratio >= threshold_close:
        print(f"🤔 Robot is between {best_name} and {second_best_neighbor}")
        print(f"🧭 Estimated {rel_position*100:.1f}% along path from {best_name} to {second_best_neighbor}")
    else:
        rel_position = 0.0
        second_best_neighbor = best_name
        print(f"🧭 Robot is confidently at {best_name}")
else:
    rel_position = 0.0
    second_best_neighbor = best_name
    print(f"🧭 Robot is confidently at {best_name}")



# === PLOTTING ===
x = list(range(len(test_signature)))

# Plot shifted and interpolated signatures
fig, ax1 = plt.subplots(figsize=(14, 6))
bar_width = 0.4

sig1 = known_signatures[best_name]
sig2 = known_signatures.get(second_best_neighbor, sig1)

interpolated_signature = [(1 - rel_position) * a + rel_position * b for a, b in zip(sig1, sig2)]

# Interpolated signature
interpolated_signature = [(1 - rel_position) * a + rel_position * b for a, b in zip(sig1, sig2)]

ax1.bar([i - bar_width/2 for i in x], test_signature, width=bar_width, label='Test Signature', alpha=0.7)
ax1.bar([i + bar_width/2 for i in x], interpolated_signature, width=bar_width, label=f'Interpolated between {best_name} and {second_best_neighbor}', alpha=0.7)

ax1.set_title("Test Signature vs Interpolated Estimated Signature")
ax1.set_xlabel("Angle Index")
ax1.set_ylabel("Distance (mm)")
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.show()

# Belief heatmap
names = list(combined_scores.keys())
scores = list(normalized_scores.values())

fig, ax2 = plt.subplots(figsize=(12, 4))
ax2.bar(names, scores, color='purple')
ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Normalized Belief")
ax2.set_title("Belief Distribution Across Signatures")
ax2.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Find best combined score
best_combined_name = max(combined_scores, key=combined_scores.get)

# Realign using combined best match
best_combined_signature = known_signatures[best_combined_name]

# Find best shift AGAIN for Euclidean or combined best
best_local_sim = -1
best_local_shift = 0
length = len(test_signature)

for shift in range(length):
    shifted = test_signature[shift:] + test_signature[:shift]
    sim = cosine_similarity(shifted, best_combined_signature)
    if sim > best_local_sim:
        best_local_sim = sim
        best_local_shift = shift

# Shifted test aligned to best_combined_signature
shifted_test_combined = test_signature[best_local_shift:] + test_signature[:best_local_shift]

# === PLOT (Per-Window Cosine) ===
x = list(range(len(shifted_test_combined)))
window_size = 5
cosine_values = []
x_vals = []

for i in range(len(shifted_test_combined) - window_size + 1):
    window_test = shifted_test_combined[i:i + window_size]
    window_match = best_combined_signature[i:i + window_size]
    sim = cosine_similarity(window_test, window_match)
    cosine_values.append(sim)
    x_vals.append(i + window_size // 2)

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(x_vals, cosine_values, color='red', marker='o')
ax.set_title(f"Cosine Similarity (Per Window) vs {best_combined_name}")
ax.set_xlabel("Angle Index")
ax.set_ylabel("Cosine Similarity")
ax.grid(True)
plt.tight_layout()
plt.show()

# === PLOT (Bar Chart Comparison) ===
fig, ax1 = plt.subplots(figsize=(14, 6))
bar_width = 0.4

ax1.bar([i - bar_width/2 for i in x], shifted_test_combined, width=bar_width, label='Shifted Test Signature', alpha=0.7)
ax1.bar([i + bar_width/2 for i in x], best_combined_signature, width=bar_width, label=f'Best Match: {best_combined_name}', alpha=0.7)

ax1.set_title(f"Aligned Test Signature vs Best Match: {best_combined_name}")
ax1.set_xlabel("Angle Index")
ax1.set_ylabel("Distance (mm)")
ax1.legend()
ax1.grid(True)
plt.tight_layout()
plt.show()

