import torch

# Train
def train(dataloader, model, loss_fn, optimizer, n_epochs=10, device="cpu"):
    size = len(dataloader.dataset)
    model.train()

    for epoch in range(n_epochs):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"epoch {epoch} \t loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

