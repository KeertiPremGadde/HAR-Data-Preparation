import numpy as np
import pandas as pd

data  = np.load("Recording1/Star_P1_S1.npy")
print(data.shape)
print(type(data.shape))
print(data[0])

index = [f'index_{num}' for num in range(data.shape[0])]
columns = [f'col_{num}' for num in range(data.shape[1])]

df = pd.DataFrame(data, columns=columns, index=index)

print(df.sample(5))