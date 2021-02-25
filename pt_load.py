
import torch
import io
import csv

with open(r'C:\Users\Bruger\Desktop\CV0_checkpoint-60', 'rb') as f:
    buffer = io.BytesIO(f.read())
state = torch.load(buffer)
with open('CV0_checkpoint.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, state.keys())
    w.writeheader()
    w.writerow(state)
