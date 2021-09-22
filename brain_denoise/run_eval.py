import torch

# Eval
def run_eval(dataloader, model, loss_fn, device="cpu"):
    model.eval()

    final_loss = 0
    n_samples = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        final_loss += loss.item()
        n_samples += len(X)

    return final_loss / n_samples
