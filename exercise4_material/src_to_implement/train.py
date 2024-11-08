import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv("data.csv", sep=';')
train_data, val_data = train_test_split(data, test_size=0.02, random_state=123)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

# hyperparameters
epochs = 50

train_dataset = ChallengeDataset(train_data, "train")
train_dataloader = t.utils.data.DataLoader(train_dataset, batch_size=8, shuffle = True, num_workers=4)

val_dataset = ChallengeDataset(val_data, "val")
val_dataloader = t.utils.data.DataLoader(val_dataset, batch_size=1, shuffle = False, num_workers=4)

# create an instance of our ResNet model
model = ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
criterion = t.nn.BCELoss()
# set up the optimizer (see t.optim)
optimizer = t.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model, criterion, optimizer, train_dataloader, val_dataloader,
                    cuda=True, early_stopping_patience=6)

# go, go, go... call fit on trainer
res = trainer.fit(epochs)
trainer.save_onnx('trained_model.onnx')

### UPLOAD ONNX FILE TO SERVER DIRECTLY XD
import requests

headers = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
    'Content-Type': 'multipart/form-data; boundary=----WebKitFormBoundarytujVLjA4xMHgP8qE',
    'Origin': 'http://lme156.informatik.uni-erlangen.de',
    'Referer': 'http://lme156.informatik.uni-erlangen.de/dl-challenge/new-job',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0',
}

files = {
    'file': open('trained_model.onnx', 'rb')
}
response = requests.post('http://lme156.informatik.uni-erlangen.de/model-upload', headers=headers, files=files, verify=False)
############################################


# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')