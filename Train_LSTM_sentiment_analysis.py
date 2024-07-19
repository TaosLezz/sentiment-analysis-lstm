import collections
import os
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import torchtext.data
import torchtext.vocab
import tqdm
from Models.LSTM_model import LSTM
import warnings
warnings.filterwarnings("ignore")

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_lstm_path = "./Weights/lstm.pt"
max_length = 256
test_size = 0.25

class Train_LSTM:
    def __init__(self, max_length = 256, test_size = 0.25):
        self.max_length = max_length
        self.test_size = test_size
    
    def get_collate_fn(self, pad_index):
        def collate_fn(batch):
            batch_ids = [i["ids"] for i in batch]
            batch_ids = nn.utils.rnn.pad_sequence(
                batch_ids, padding_value=pad_index, batch_first=True
            )
            batch_length = [i["length"] for i in batch]
            batch_length = torch.stack(batch_length)
            batch_label = [i["label"] for i in batch]
            batch_label = torch.stack(batch_label)
            batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
            return batch

        return collate_fn
    
    def get_data_loader(self, dataset, batch_size, pad_index, shuffle=False):
        collate_fn = self.get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
        return data_loader
    
    def tokenize_example(self, example, tokenizer, max_length):
        tokens = tokenizer(example["text"])[:max_length]
        length = len(tokens)
        return {"tokens": tokens, "length": length}
    
    def numericalize_example(self, example, vocab):
        ids = vocab.lookup_indices(example["tokens"])
        return {"ids": ids}
    
    def count_parameters(self, model):   
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                elif "weight" in name:
                    nn.init.orthogonal_(param)
    
    def train(self, dataloader, model, criterion, optimizer, device):
        model.train()
        epoch_losses = []
        epoch_accs = []
        for batch in tqdm.tqdm(dataloader, desc="training..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
        return np.mean(epoch_losses), np.mean(epoch_accs)

    def get_accuracy(prediction, label):
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy

    def predict_sentiment(text, model, tokenizer, vocab, device):
        tokens = tokenizer(text)
        ids = vocab.lookup_indices(tokens)
        length = torch.LongTensor([len(ids)])
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        prediction = model(tensor, length).squeeze(dim=0)
        probability = torch.softmax(prediction, dim=-1)
        predicted_class = prediction.argmax(dim=-1).item()
        predicted_probability = probability[predicted_class].item()
        return predicted_class, predicted_probability

train_model = Train_LSTM(max_length, test_size)
train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])

tokenizer = torchtext.data.utils.get_tokenizer("basic_english")


train_data = train_data.map(
    train_model.tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)
test_data = test_data.map(
    train_model.tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)


train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]
min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)
unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]
vocab.set_default_index(unk_index)
train_data = train_data.map(train_model.numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(train_model.numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(train_model.numericalize_example, fn_kwargs={"vocab": vocab})
train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label", "length"])
test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])
vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = len(train_data.unique("label"))
n_layers = 2
bidirectional = True
dropout_rate = 0.5
# DÙng 'ids', 'length', 'label' để train
batch_size = 512

train_data_loader = train_model.get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = train_model.get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = train_model.get_data_loader(test_data, batch_size, pad_index)
model = LSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout_rate,
    pad_index,
)
print(f"The model has {train_model.count_parameters(model):,} trainable parameters")
model.apply(train_model.initialize_weights)
# vectors = torchtext.vocab.GloVe()
# pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
# model.embedding.weight.data = pretrained_embedding

lr = 5e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

#Load model
model.load_state_dict(torch.load(weights_lstm_path, map_location=torch.device('cpu')))

n_epochs = 10
best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train_model.train(
        train_data_loader, model, criterion, optimizer, device
    )
    valid_loss, valid_acc = train_model.evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "lstm.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")



    
    

