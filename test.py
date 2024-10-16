import torch
from torch.utils.data import DataLoader, Dataset

# Custom dataset example
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Example data
data = [i for i in range(100)]
dataset = MyDataset(data)

# Create a data loader
loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)

# Example training loop using the loader
for batch in loader:
    print(batch)  # This will print batches of data
