from functools import partial
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import os, glob, math, datetime, csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

class SentimentDataset(Dataset):
    def __init__(self, statements, labels, tokenizer, max_length=1000):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, idx):
        statement = self.statements[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(statement)
        if len(tokens) > self.max_length:
            i_start = torch.randint(low=0, high=len(tokens) - self.max_length + 1, size=(1, )).item()
            tokens = tokens[i_start:i_start+self.max_length]
        tokens = torch.tensor(tokens)

        return tokens, torch.tensor(label)

def collate_fn(batch):
    tokens, labels = zip(*batch)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return tokens_padded, labels


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, nhead, num_encoder_layers, num_classes, dropout, dim_feedforward):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)
        self.positional_embedding = nn.Embedding(max_length, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True, dropout=dropout, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
        # self.batch_norm = nn.BatchNorm1d(d_model)
    
    def forward(self, src):
        src_positions = torch.arange(0, src.size(1), device=src.device).unsqueeze(0).expand(src.size(0), -1)
        # src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.embedding(src) + self.positional_embedding(src_positions)
        # src = self.pos_encoder(src)
        # src = self.batch_norm(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Global average pooling
        output = self.fc(output)
        return output

def tokenization(text, tokenizer):
    return tokenizer.encode(text).ids


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    DROPOUT = 0.2
    BATCH_SIZE = 64
    D_MODEL = 16
    FF_SIZE = 8
    EPOCH = 50
    
    file = glob.glob(os.path.expanduser("~/Documents/projects/chatgpt-from-scratch/data/*.csv"))[0]
    df = pd.read_csv(file, index_col=0).dropna(how="any", axis=0)
    
    threshold = df.groupby("status").count().quantile(0.7)
    for status in df["status"].unique():
        _data = df[df["status"] == status]
        if len(_data) < threshold.values[0]:
            print(f"{status}, {len(_data)}")
            n = threshold.values[0] // len(_data)
            for _ in range(int(n)):
                df = pd.concat((df, _data))
                
    print(f"data shape: {df.shape}")
    
    texts = df["statement"].tolist()
    
    vocab_size = 30000
    # tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    # trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=5, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    # tokenizer.pre_tokenizer = Whitespace()
    # tokenizer.train_from_iterator(texts, trainer=trainer)
    print(f"Load tokenizer...")
    tokenizer = Tokenizer.from_file("data/tokenizer-mental-health.json")
            
    max_length = int(df["statement"].apply(len).quantile(0.9))
    
    print(f"vocab size is: {vocab_size}")
    
    temp_set = set()
    for item in tqdm(tokenizer.encode_batch(texts), desc="compute vocab size"):
        temp_set = temp_set | set(item.ids)    
    
    statements = df["statement"].values
    labels = df["status"].values

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    train_statements, val_statements, train_labels, val_labels = train_test_split(statements, encoded_labels, test_size=0.2, random_state=42)
    # Create datasets
    train_dataset = SentimentDataset(train_statements, train_labels, tokenizer=partial(tokenization, tokenizer=tokenizer), max_length=max_length)
    val_dataset = SentimentDataset(val_statements, val_labels, tokenizer=partial(tokenization, tokenizer=tokenizer), max_length=max_length)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Instantiate the model
    model = CustomTransformerModel(vocab_size=vocab_size, 
                                   d_model=D_MODEL, 
                                   max_length=max_length,
                                   nhead=2, 
                                   num_encoder_layers=4, 
                                   num_classes=len(label_encoder.classes_), 
                                   dropout=DROPOUT, 
                                   dim_feedforward=FF_SIZE).to(device)
    
    # model_state_dict = torch.load("model_state.pth", weights_only=True)
    # model.load_state_dict(model_state_dict)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-8)

    # Training loop
    num_epochs = EPOCH
    model.train()
    
    write = open("epoches.csv", "w")
    writer = csv.writer(write)
    writer.writerow([f"Dropout: {DROPOUT}, Batch_size: {BATCH_SIZE}, D_model: {D_MODEL}, Feed forward: {FF_SIZE}"])
    
    try:
        for epoch in range(num_epochs):
            print(f"{datetime.datetime.now().strftime('%H:%M:%S %p')}: start training epoch {epoch+1}...")
            total_losses = 0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_losses += loss.item()                
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            training_loss = total_losses/len(train_loader)
            train_accuracy = 100 * correct / total
                  
            # Optional: Evaluate on the validation set after each epoch
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)        
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}, Training loss: {training_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%; Validation Loss: {val_loss/len(val_loader):.6f}, Accuracy: {100 * correct / total:.2f}%, Learning rate: {scheduler.get_last_lr()[0]}')
            writer.writerow([f'Epoch {epoch+1}, Training loss: {training_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%; Validation Loss: {val_loss/len(val_loader):.6f}, Accuracy: {100 * correct / total:.2f}%, Learning rate: {scheduler.get_last_lr()[0]}'])
            model.train()
    except KeyboardInterrupt as e:
        raise e
    finally:
        write.close()
        torch.save(model.state_dict(), "model_state.pth")