import collections
import os
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torchtext.data
import torchtext.vocab
import tqdm
from Models.LSTM_model import LSTM
from datasets import load_from_disk
import time
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./Weights/lstm.pt"
data_train_path = "./Data/imdb/train"
data_test_path = "./Data/imdb/test"
max_length = 256
test_size = 0.25
data_files = {'train':'train-00000-of-00001.parquet',
              'test':'test-00000-of-00001.parquet'}

class SentimentAnalyzer:
    def __init__(self, max_length, test_size, model_path, data_train_path, data_test_path, data_files):
        self.max_length = max_length
        self.test_size = test_size
        self.model_path = model_path
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.data_files = data_files
        self.tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        
        # self.train_model = Train_LSTM(max_length, test_size)
        self._prepare_data()
        self._build_vocab()
        self._init_model()
    
    def _prepare_data(self):
        train_data, test_data = datasets.load_dataset('parquet', data_dir=r'E:\aHieu\pytorch\sentiment-analysis\Data1\imdb\plain_text', 
                                                      data_files=self.data_files, split=["train", "test"])
        
        train_data = train_data.map(
            self.tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": self.max_length}
        )
        test_data = test_data.map(
            self.tokenize_example, fn_kwargs={"tokenizer": self.tokenizer, "max_length": self.max_length}
        )
        
        train_valid_data = train_data.train_test_split(test_size=self.test_size)
        self.train_data = train_valid_data["train"]
        self.valid_data = train_valid_data["test"]
    
    def _build_vocab(self):
        min_freq = 5
        special_tokens = ["<unk>", "<pad>"]
        
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            self.train_data["tokens"],
            min_freq=min_freq,
            specials=special_tokens,
        )
        self.unk_index = self.vocab["<unk>"]
        self.pad_index = self.vocab["<pad>"]
        self.vocab.set_default_index(self.unk_index)
    
    def _init_model(self):
        vocab_size = len(self.vocab)
        embedding_dim = 300
        hidden_dim = 300
        output_dim = len(self.train_data.unique("label"))
        n_layers = 2
        bidirectional = True
        dropout_rate = 0.5
        
        self.model = LSTM(
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout_rate,
            self.pad_index,
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        # print(f"The model has {self.count_parameters(self.model):,} trainable parameters")
    
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

    def get_accuracy(self, prediction, label):
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy

    def predict_sentiment(self, text, model, tokenizer, vocab, device):
        tokens = tokenizer(text)
        ids = vocab.lookup_indices(tokens)
        length = torch.LongTensor([len(ids)])
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
        prediction = model(tensor, length).squeeze(dim=0)
        probability = torch.softmax(prediction, dim=-1)
        predicted_class = prediction.argmax(dim=-1).item()
        predicted_probability = probability[predicted_class].item()
        return predicted_class, predicted_probability
    
    def result_predict_sentiment(self, text):
        return self.predict_sentiment(text, self.model, self.tokenizer, self.vocab, device)


# sentiment_analysis = SentimentAnalyzer(max_length, test_size, model_path, data_train_path, data_test_path)


# text = "This film is terrible!"

# text1 = "This film is good!"
# print(sentiment_analysis.result_predict_sentiment(text1))


example_sentences = [
    "I love this product! It's amazing.",
    "This is the worst experience I've ever had.",
    "I'm not sure if I like this or not.",
    "The quality of the item is fantastic.",
    "I will never buy this again."
]


st.title("Sentiment Analysis Application")

example_choice = st.selectbox("Chọn một câu ví dụ:", [" "] + example_sentences)

if example_choice:
    st.query_params["user_input"] = example_choice

user_input = st.query_params.get("user_input", '')

user_input = st.text_input("Text: ", user_input)

if st.button("Dự đoán"):
    if user_input:
        sentiment_analysis = SentimentAnalyzer(max_length, test_size, model_path, data_train_path, data_test_path, data_files)
        predict, probability = sentiment_analysis.result_predict_sentiment(user_input)
        result = 'Positive' if predict == 1 else 'Negative'
        st.write(f"Kết quả dự đoán: {result}  \nScore: {probability}")
    else:
        st.write("Vui lòng nhập dữ liệu")

