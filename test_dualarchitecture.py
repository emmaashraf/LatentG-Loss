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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
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


def evaluate_set(model,set_loader, DEVICE):
    model.eval()

    y_pred_labels = []
    y_true_labels = []

    with torch.no_grad():
        for X,y in set_loader :
            X, y = X.to(DEVICE).unsqueeze(1), y.to(DEVICE).squeeze(1)
            y_pred, _, _ = model(X)

            predicted_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
            true_labels = y.cpu().numpy()

            y_pred_labels.extend(predicted_labels)
            y_true_labels.extend(true_labels)
    
    test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
    test_precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    test_recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    test_f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    print(f'Accuracy: {test_accuracy:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall: {test_recall:.4f}')
    print(f'F1 Score: {test_f1:.4f}')

def main():
    MODEL_SAVE_PATH = "C:/Users/Korhan/Desktop/workspace/vsCodeWorkspace/Python_Workspace/mental_health_sentiment_analysis/latentG_loss_best_dualtextcnnModel.pt"
    BEST_MODEL_SAVE_PATH = "C:/Users/Korhan/Desktop/workspace/vsCodeWorkspace/Python_Workspace/mental_health_sentiment_analysis/best_dualtextcnnModel.pt"
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    BATCH_SIZE = 16
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
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device(DEVICE)))
    
    print("Latest Model Results :")
    print("On Test Set :")
    evaluate_set(model=model, set_loader=test_dataloader, DEVICE=DEVICE)
    print("On Train Set :")
    evaluate_set(model=model, set_loader=train_dataloader, DEVICE=DEVICE)

    bestmodel = DualTextCNN(input_dim=300, num_classes=7, latent_dim=32).to(DEVICE)
    bestmodel.load_state_dict(torch.load(BEST_MODEL_SAVE_PATH, weights_only=True))
    bestmodel.eval()
    
    print("Best Model Results :")
    print("On Test Set: ")
    evaluate_set(model=bestmodel, set_loader=test_dataloader, DEVICE=DEVICE)
    print("On Train Set: ")
    evaluate_set(model=bestmodel, set_loader=train_dataloader, DEVICE=DEVICE)

if __name__ == "__main__":
    main()