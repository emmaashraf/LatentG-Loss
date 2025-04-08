from copy import deepcopy
import torch 
import torch.nn as nn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import string
import nltk
from torch import optim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import plotly.figure_factory as ff
from textblob import TextBlob
import numpy as np
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from torch.utils.data import Dataset, DataLoader
from TextDataset import TextDataset
from DualArchitecture import DualTextCNN


def main():
    MODEL_SAVE_PATH = "C:/Users/Korhan/Desktop/workspace/vsCodeWorkspace/Python_Workspace/mental_health_sentiment_analysis/newdualtextcnnModel.pt"
    BEST_MODEL_SAVE_PATH = "C:/Users/Korhan/Desktop/workspace/vsCodeWorkspace/Python_Workspace/mental_health_sentiment_analysis/best_newdualtextcnnModel.pt"
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    LEARNING_RATE = 0.01
    BATCH_SIZE = 64
    EPOCHS = 500
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    X_train_embeddings = np.load("X_train_embeddings.npy")
    y_train_encoded = np.load("y_train_encoded.npy")
    X_test_embeddings = np.load("X_test_embeddings.npy")
    y_test_encoded = np.load("y_test_encoded.npy")
    
    X_train_embeddings = np.array(X_train_embeddings)
    X_test_embeddings = np.array(X_test_embeddings)

    y_train_encoded = np.array(y_train_encoded)
    y_test_encoded = np.array(y_test_encoded)

    y_train_encoded = np.reshape(y_train_encoded, (-1,))
    y_test_encoded = np.reshape(y_test_encoded, (-1,))


    train_dataset = TextDataset(X_train_embeddings, y_train_encoded)
    test_dataset = TextDataset(X_test_embeddings, y_test_encoded)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DualTextCNN(input_dim=300, num_classes=7, latent_dim=32).to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9998)

    best_loss = 100
    best_model = deepcopy(model.state_dict())
    best_model_results = {"train_loss" : 100.0,
                          "test_loss" : 100.0}
    

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, (X,y) in enumerate(tqdm(train_dataloader)):
            X = X.to(DEVICE).unsqueeze(1)
            y = y.to(DEVICE).squeeze(1)
            class_pred, reconstructed_pred, _ = model(X)
            
            loss1 = criterion1(class_pred, y)
            loss2 = criterion2(reconstructed_pred, X)
            total_loss = loss1 + loss2 * 90
            train_running_loss += total_loss.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        train_loss = train_running_loss / (idx+1)

        model.eval()
        test_running_loss = 0
        with torch.no_grad():
            for idx, (X,y) in enumerate(tqdm(test_dataloader)):
                X = X.to(DEVICE).unsqueeze(1)
                y = y.to(DEVICE).squeeze(1)
                class_pred, reconstructed_pred, _ = model(X)

                loss1 = criterion1(class_pred, y)
                loss2 = criterion2(reconstructed_pred, X)
                total_loss = loss1 + loss2 * 90
                test_running_loss += total_loss
            
            test_loss = test_running_loss / (idx+1)
        
        print(f" EPOCH : {epoch + 1} | Train Loss : {train_loss:.4f} ")
        print(f" EPOCH : {epoch + 1} | Test Loss : {test_loss:.4f} ")
        print("-"*50)

        if test_loss < best_loss :
            best_model = deepcopy(model.state_dict())
            best_model_results["train_loss"] = train_loss
            best_model_results["test_loss"] = test_loss
            best_loss = test_loss
        
        scheduler.step()
        torch.cuda.empty_cache()

    print("Saving the latest model...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    torch.save(best_model, BEST_MODEL_SAVE_PATH)    




if __name__ == "__main__": 
    main()