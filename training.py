import torch
import torch.nn as nn
import torch.optim as optim

from Flattener import Flattener
from helpers import compute_accuracy

nn_model = nn.Sequential(
    Flattener(),
    nn.Linear(3 * 32 * 32, 100),
    nn.ReLU(inplace=True),
    nn.Linear(100, 10)
)
nn_model.type(torch.FloatTensor)

loss = nn.CrossEntropyLoss().type(torch.FloatTensor)
optimizer = optim.SGD(nn_model.parameters(), lr=1e-2, weight_decay=1e-1)

loss_history = []
train_history = []
val_history = []

num_epoch = 3

for epoch in range(num_epoch):
    nn_model.train()

    loss_acc = 0
    correct_samples = 0
    total_samples = 0
    for i_step, (x, y) in enumerate(train_loader):
        prediction = nn_model(x)
        loss_value = loss(prediction, y)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        _, indices = torch.max(prediction, 1)
        correct_samples += y.shape[0]
        loss_acc += loss_value
        ave_loss = loss_acc / (i_step + 1)

        train_accuracy = float(correct_samples) / total_samples
        val_accuracy = compute_accuracy(nn_model, val_loader)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        val_history.append(val_accuracy)

        print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))
