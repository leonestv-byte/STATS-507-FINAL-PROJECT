from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer
import numpy as np


results = {
	"egy": {
		"linear_nn": 0.2561383928571428,
		"arat5v2": 0.26944196428571426
	},
	"glf": {
		"linear_nn": 0.36966517857142855,
		"arat5v2": 0.38546875
	},
	"lev": {
		"linear_nn": 0.28439732142857144,
		"arat5v2": 0.29667410714285714
	},
	"mgr": {
		"linear_nn": 0.3828794642857143,
		"arat5v2": 0.39872767857142855
	}
}


df = pd.DataFrame(results).T 


ax = df.plot(kind="bar", figsize=(8, 5))
plt.title("Model Performance by Dialect")
plt.xlabel("Dialect")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# egy_ds = load_dataset("QCRI/arabic_pos_dialect", "egy")
# glf_ds = load_dataset("QCRI/arabic_pos_dialect", "glf") 
# lev_ds = load_dataset("QCRI/arabic_pos_dialect", "lev") 
mgr_ds = load_dataset("QCRI/arabic_pos_dialect", "mgr") 

dialect_ds = mgr_ds


df = dialect_ds['train'].to_pandas()


dictionary = {}

def word_bucket(df):
	for x in df['words']:
		for word in x:
			count = dictionary.get(word, 0)
			dictionary[word] = count + 1

word_bucket(df)



word_counts = dictionary
dictionary.pop("EOS")

sorted_items = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
top_10 = sorted_items[:10]

words = [w for w, _ in top_10]
counts = [c for _, c in top_10]

plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.xticks(rotation=45, ha='right')
plt.title("MGR Word Counts")
plt.xlabel("Word")
plt.ylabel("Count")
plt.tight_layout()
plt.show()













