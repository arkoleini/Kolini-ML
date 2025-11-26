import os
import pandas as pd

# Try to read local CSV first; if missing, fall back to sklearn's diabetes dataset
csv_path = 'diabetes.csv'
if os.path.exists(csv_path):
	diabetes_df = pd.read_csv(csv_path)
	print(f"Loaded '{csv_path}' from disk.\n")
else:
	# Lazy import scikit-learn only when needed
	from sklearn import datasets

	print(f"File '{csv_path}' not found. Loading diabetes dataset from sklearn and writing a copy to '{csv_path}'.\n")
	data = datasets.load_diabetes()
	# Build a DataFrame with feature names if available
	try:
		feature_names = data.feature_names
	except AttributeError:
		# older scikit-learn versions might not include feature_names
		feature_names = [f'feature_{i}' for i in range(data.data.shape[1])]

	diabetes_df = pd.DataFrame(data.data, columns=feature_names)
	# Add target column
	diabetes_df['target'] = data.target
	# Save a copy so subsequent runs can use the CSV
	diabetes_df.to_csv(csv_path, index=False)

print(diabetes_df.head())