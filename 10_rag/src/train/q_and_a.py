import torch.optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

def train(
        model,
        train_dataset: Dataset,
        test_dataset: Dataset,
        batch_size = 10,
        learning_rate = 1e-5,
        epochs = 3,
        device = "cpu"
):
    model.to(device)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / 1000))

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, labels = batch["input_ids"].to(device)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

        # Evaluate the model on the test set
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch["input_ids"].to(device)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(**inputs)
                loss = outputs.loss
                total_loss += loss.item()

        print(f"Test Loss: {total_loss / len(test_loader)}")
