import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

checkpoints_folder = "checkpoints"
from model.lora import LoraLinear
def train(
    model,
    train_dataset,
    eval_dataset,
    num_epochs=3,
    learning_rate=5e-5,
    batch_size=8,
    device="cuda",
):
    print("Iniciando entrenamiento...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.to(device)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    for layer in model.model.layers:
        if hasattr(layer, 'self_attn'):
            layer.self_attn.q_proj = LoraLinear(layer.self_attn.q_proj, r=16)
            layer.self_attn.k_proj = LoraLinear(layer.self_attn.k_proj, r=16)
            layer.self_attn.v_proj = LoraLinear(layer.self_attn.v_proj, r=16)
            layer.self_attn.o_proj = LoraLinear(layer.self_attn.o_proj, r=16)
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        print(f"\n--- Época {epoch + 1}/{num_epochs} ---")

        for batch_idx, batch in enumerate(train_dataloader):
            x, _ = batch
            x = {k: v.to(device) for k, v in x.items()}
            outputs = model(**x)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_description(f"Época {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Fin de Época {epoch + 1}: Pérdida de Entrenamiento Promedio = {avg_train_loss:.4f}")

        model.eval()
        total_eval_loss = 0
        print(f"\nEvaluando al final de la época {epoch + 1}...")
        with torch.no_grad():
            for eval_batch in tqdm(eval_dataloader, desc="Evaluación"):
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                outputs = model(**eval_batch)
                total_eval_loss += outputs.loss.item()
        avg_eval_loss = total_eval_loss / len(eval_dataloader)
        print(f"Fin de Época {epoch + 1}: Pérdida de Validación Promedio = {avg_eval_loss:.4f}")
        # Guardar el modelo
        model.save_pretrained(f"{checkpoints_folder}/lora_model_epoch_{epoch + 1}.ckpt")

    progress_bar.close()
    print("Entrenamiento completado.")