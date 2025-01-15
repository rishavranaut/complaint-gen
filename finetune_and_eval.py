import os
import pickle
import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)

class VQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        image = self.dataset[idx]['image']  
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(answer, max_length=256, pad_to_max_length=True, return_tensors='pt')
        encoding["labels"] = labels
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        return encoding

dataset = load_dataset("cerelac2/consumer-complaint-vqa")
train_dataset = dataset['train']
valid_dataset = dataset['validation']
train_dataset = VQADataset(dataset=train_dataset, processor=processor)
valid_dataset = VQADataset(dataset=valid_dataset, processor=processor)
batch_size = 12
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)
num_epochs = 10
patience = 10
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_masked = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    model.eval()
    eval_loss = 0
    for batch in tqdm(valid_dataloader, desc=f"Validating epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        attention_masked = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()
    tracking_information.append((epoch_loss / len(train_dataloader), eval_loss / len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print(f"Epoch {epoch+1} - Training loss: {epoch_loss / len(train_dataloader)} - Eval Loss: {eval_loss / len(valid_dataloader)} - LR: {optimizer.param_groups[0]['lr']}")
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained("Model/blip-saved-model")
        print("Saved model to Model/blip-saved-model")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            print("Early stopping triggered")
            break

# Save training information
pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("Finetuning complete!")

class VQATestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        image = self.dataset[idx]['image']
        if isinstance(image, Image.Image):
            image = image.convert("RGB")  
        else:
            image = Image.open(image).convert("RGB")
        encoding = self.processor(images=image, text=question, return_tensors="pt", padding="max_length", truncation=True)
        return encoding, answer
test_dataset = VQATestDataset(dataset['test'],processor)
finetuned_model = BlipForQuestionAnswering.from_pretrained("Model/blip-saved-model")
predicted_answers = []
ground_truths = []
print("Evaluating on Test Set:")
finetuned_model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for idx in tqdm(range(len(test_dataset))):
        inputs, ground_truth = test_dataset[idx]
        outputs = finetuned_model.generate(**inputs,max_new_tokens=128)
        predicted_answer = processor.decode(outputs[0], skip_special_tokens=True)
        predicted_answers.append(predicted_answer)
        ground_truths.append(ground_truth)