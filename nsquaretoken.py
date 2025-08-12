import torch
import spacy
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import pandas as pd

# HYPERPARAMETERS
learning_rate = 0.001
# linear layer size
#size1 = 384  # THIS IS FIXED
size1 = 300
size2 = 500
test_train_ratio = 0.8

#load our excel dataset
dataset = pd.read_csv("Database/target_dataset_2.csv")
data_length = len(dataset)
print(data_length, "Total rows")

#add in our dataset split into training and test data
class KeywordDataset(Dataset):
    """Keyword dataset."""

    def __init__(self, csv_file, transform=None, indices=None):
        """
        Arguments:
            csv_file (string): Path to the csv file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.keyword_df = pd.read_csv(csv_file)
        if indices:
            self.keyword_df = self.keyword_df.iloc[indices, :].reset_index()
        self.transform = transform

    def __len__(self):
        return len(self.keyword_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        phrase = self.keyword_df.loc[idx, "Phrase"]
        target = self.keyword_df.loc[idx, "target"]

        if self.transform:
            phrase = self.transform(phrase)

        return phrase, target

class To_Tensor:
    def __init__(self,dtype=torch.float32,model="en_core_web_md"):
        self.dtype=dtype
        self.model=model
    def __call__(self,text):
        nlp = spacy.load(self.model) #change back later
        tokenized_text = nlp(text)
        token_vector = [token.vector for token in tokenized_text]
        token_tensor = torch.tensor(token_vector,dtype=self.dtype)
        return token_tensor

#pad tensors
def collate_fn(batch):
    t1, t2 = zip(*batch)
    #t1 in the tensor has variable lengths
    padded_t1 = pad_sequence(t1)
    #t2 can be stacked directly
    t2 = [int(i) for i in t2]
    t2 = [torch.tensor(i) for i in t2]
    stacked_t2 = torch.stack(t2)
    return padded_t1,stacked_t2

#transform = SentenceTransform()
transform = To_Tensor()
train_size = int(test_train_ratio * len(dataset))
training_data = KeywordDataset("Database/target_dataset_2.csv", transform=transform, indices=range(train_size))
test_data = KeywordDataset("Database/target_dataset_2.csv", transform=transform, indices=range(train_size, len(dataset)))

#change later
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

#accelerator
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

#neural network model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(size1, size2),
            torch.nn.ReLU(),
            torch.nn.Linear(size2, size2),
            torch.nn.ReLU(),
            torch.nn.Linear(size2, 1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        x=x.mean(dim=0)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = NeuralNetwork()
model.to(device)

#loss function
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device).float().unsqueeze(1)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float().unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((torch.sigmoid(pred) >= 0.5).float() == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
print("Complete.")

test_loop(test_dataloader, model, loss_fn)
