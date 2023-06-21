import pandas as pd

# Initialize an empty list
chunks = []

# Iterate over the files and append chunks to the list
for i in range(18, 23):
    filename = f"valeursfoncieres-20{i}.txt"
    chunk = pd.read_csv(filename, sep='|', header=0, low_memory=False)
    chunks.append(chunk)

# Concatenate the chunks into a single DataFrame
df = pd.concat(chunks)

print(df.shape)
