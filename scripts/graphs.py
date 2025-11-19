from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd


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


egy_ds = load_dataset("QCRI/arabic_pos_dialect", "egy")
# glf_ds = load_dataset("QCRI/arabic_pos_dialect", "glf") 
# lev_ds = load_dataset("QCRI/arabic_pos_dialect", "lev") 
# mgr_ds = load_dataset("QCRI/arabic_pos_dialect", "mgr") 


egy_ds["words"]

