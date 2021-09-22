import torch
from run_eval import run_eval
import numpy as np

# Train
def train(
    dataloader,
    model,
    loss_fn,
    optimizer,
    n_epochs=10,
    validloader=None,
    testloader=None,
    device="cpu",
):
    size = len(dataloader.dataset)
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_samples = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_samples += len(X)

        # Run eval
        if validloader is not None:
            val_loss = run_eval(validloader, model, loss_fn)
        else:
            val_loss = np.nan

        # Run test
        if testloader is not None:
            test_loss = run_eval(testloader, model, loss_fn)
        else:
            test_loss = np.nan

        print(
            f"epoch {epoch} \t loss: {epoch_loss:>7f} \t eval: {val_loss:>7f} \t test: {test_loss:>7f} [{epoch:>5d}/{n_epochs:>5d}]"
        )
