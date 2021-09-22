import torch
import numpy as np

# Optimization: one epoch
def run_epoch(
    dataloader, 
    model, 
    loss_fn, 
    device="cpu", 
    train=True,
    optimizer=None,
    n_epochs=10,
    ):
    if train:
        model.train()
    else:
        model.eval()

    final_loss = 0
    n_samples = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # Backpropagation
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Monitor loss
        final_loss += loss
        n_samples += len(X)

    return final_loss / n_samples


# Optimization: all epochs
def train_eval_model(
    train_loader,
    model,
    loss_fn,
    optimizer,
    n_epochs=10,
    valid_loader=None,
    test_loader=None,
    device="cpu",
):
    for epoch in range(n_epochs):
        # Train set
        train_loss = run_epoch(
            dataloader=train_loader, 
            model=model, 
            loss_fn=loss_fn, 
            device=device, 
            train=True,
            optimizer=optimizer,
            n_epochs=n_epochs
        )

        # Validation set
        if valid_loader is not None:
            val_loss = run_epoch(
                dataloader=valid_loader, 
                model=model, 
                loss_fn=loss_fn, 
                device=device, 
                train=False,
                optimizer=None,
                n_epochs=n_epochs
            )
        else:
            val_loss = np.nan

        # Test set
        if test_loader is not None:
            test_loss = run_epoch(
                dataloader=test_loader, 
                model=model, 
                loss_fn=loss_fn, 
                device=device, 
                train=False,
                optimizer=None,
                n_epochs=n_epochs
            )        
        else:
            test_loss = np.nan

        print(
            f"epoch {epoch} \t loss: {train_loss:>7f} \t eval: {val_loss:>7f} \t test: {test_loss:>7f} [{epoch:>5d}/{n_epochs:>5d}]"
        )
