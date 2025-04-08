from copy import deepcopy
from scipy.stats import multivariate_normal
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
from TextCNN import TextCNN
from sklearn.mixture import GaussianMixture

def proba_comb_features(gmm, comb_pred_vectors):
    if comb_pred_vectors is None :
        return None
    predictions_gmm = gmm.predict(comb_pred_vectors)
    pred_probs_gmm = gmm.predict_proba(comb_pred_vectors)

    component = predictions_gmm[0]
    mean = gmm.means_[component]
    covariance = gmm.covariances_[component]
    
    pdf = multivariate_normal.pdf(comb_pred_vectors, mean=mean, cov=covariance)
    return pdf


def main():
    MODEL_SAVE_PATH = "C:/Users/Korhan/Desktop/workspace/vsCodeWorkspace/Python_Workspace/mental_health_sentiment_analysis/latentG_loss_dualtextcnnModel.pt"
    BEST_MODEL_SAVE_PATH = "C:/Users/Korhan/Desktop/workspace/vsCodeWorkspace/Python_Workspace/mental_health_sentiment_analysis/latentG_loss_best_dualtextcnnModel.pt"
    PRETRAINED_MODEL_PATH = "C:/Users/Korhan/Desktop/workspace/vsCodeWorkspace/Python_Workspace/mental_health_sentiment_analysis/best_newdualtextcnnModel.pt"

    torch.cuda.empty_cache()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    LEARNING_RATE = 0.01
    BATCH_SIZE = 128
    EPOCHS = 150
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALPHA = 0.5
    BETA = 0.5
    MAX_LIM = 3.0
    MIN_LIM = 0.0

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

    print("Data is ready.")

    teacher_model = DualTextCNN(input_dim=300, num_classes=7, latent_dim=32).to(DEVICE)
    teacher_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device(DEVICE)))
    print("Teacher model has been loaded succesfully.")

    teacher_model.eval()
    teacher_latent_embeddings = []
    teacher_pred_logits = []
    with torch.no_grad():
        for X, y in train_dataloader:
            X, y = X.to(DEVICE).unsqueeze(1), y.to(DEVICE).squeeze(1)
            y_pred, rec_pred, teacher_latent_embedding_batch = teacher_model(X)
            teacher_pred_logits.append(y_pred)
            teacher_latent_embeddings.append(teacher_latent_embedding_batch)
    teacher_latent_embeddings = torch.cat(teacher_latent_embeddings, dim=0)
    teacher_pred_logits = torch.cat(teacher_pred_logits, dim=0)
    teacher_combined_features = torch.cat( (teacher_latent_embeddings, teacher_pred_logits), dim=-1)
    teacher_combined_features = teacher_combined_features.detach().cpu().numpy()
    print("Teacher's Combined Features are ready.")


    print("Fitting a Gaussian Mixture Model to Combined Features...")
    param_grid = {
    'n_components': [7],
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'max_iter': [100, 200, 500],
    }

    gmm = GaussianMixture()
    grid_search = GridSearchCV(gmm, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(teacher_combined_features)

    best_gmm_model = grid_search.best_estimator_
    print("GMM fitted. Here are the results : ")
    print("Best Model: ", best_gmm_model)
    print("Gaussian Means Shape:", best_gmm_model.means_.shape)
    print("Covariances Shape:", best_gmm_model.covariances_.shape)


    print("Starting to the training of student model...")
    student_model = DualTextCNN(input_dim=300, num_classes=7, latent_dim=32).to(DEVICE)

    optimizer = optim.SGD(student_model.parameters(), lr=LEARNING_RATE)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9998)

    best_loss = 100
    best_model = deepcopy(student_model.state_dict())
    best_model_results = {"train_loss" : 100.0,
                          "test_loss" : 100.0}
    
    for epoch in tqdm(range(EPOCHS)):
        student_model.train()
        train_running_loss = 0
        for idx, (X,y) in enumerate(tqdm(train_dataloader)):
            X = X.to(DEVICE).unsqueeze(1)
            y = y.to(DEVICE).squeeze(1)
            class_pred, reconstructed_pred, student_latent_vector_batch = student_model(X)

            classification_loss = criterion1(class_pred, y)
            reconstruction_loss = criterion2(reconstructed_pred, X)

            comb_pred_vectors = torch.cat((student_latent_vector_batch , class_pred), dim=-1)
            batch_data = teacher_pred_logits[idx* BATCH_SIZE : idx*BATCH_SIZE+BATCH_SIZE]

            euclid_dist = torch.sqrt(torch.sum((class_pred - batch_data) ** 2, dim=1))
            mean_distance = euclid_dist.mean()
            std_distance = euclid_dist.std()
            euclid_dist = (euclid_dist - mean_distance) / std_distance
            euclid_dist = euclid_dist.sum().cpu().detach().numpy()

            comb_pred_vectors = comb_pred_vectors.cpu().detach().numpy()
            prob = proba_comb_features(gmm=best_gmm_model, comb_pred_vectors=comb_pred_vectors)
            latentG_loss =  ( ALPHA * ( (1 - prob).sum())  +  BETA * (euclid_dist) )
            latentG_loss = torch.from_numpy(np.array(latentG_loss)).to(DEVICE)
            latentG_loss = classification_loss * (1 + (epoch / EPOCHS) * latentG_loss)  +  reconstruction_loss * 75
            latentG_loss = torch.clamp(latentG_loss, min=MIN_LIM, max=MAX_LIM)
            train_running_loss += latentG_loss.item()

            optimizer.zero_grad()

            latentG_loss.backward()

            optimizer.step()
        
        train_loss = train_running_loss / (idx+1)

        student_model.eval()
        test_running_loss = 0
        with torch.no_grad():
            for idx, (X,y) in enumerate(tqdm(test_dataloader)):
                X = X.to(DEVICE).unsqueeze(1)
                y = y.to(DEVICE).squeeze(1)
                class_pred, reconstructed_pred, student_latent_vector_batch = student_model(X)
                classification_loss = criterion1(class_pred, y)
                reconstruction_loss = criterion2(reconstructed_pred, X)
                total_test_loss = classification_loss + reconstruction_loss * 75
                test_running_loss += total_test_loss.item()
            test_loss = test_running_loss / (idx+1)
        print(f" EPOCH : {epoch + 1} | Train Loss : {train_loss:.4f} ")
        print(f" EPOCH : {epoch + 1} | Test Loss : {test_loss:.4f} ")
        print("-"*50)

        if test_loss < best_loss :
            best_model = deepcopy(student_model.state_dict())
            best_model_results["train_loss"] = train_loss
            best_model_results["test_loss"] = test_loss
            best_loss = test_loss
            
        scheduler.step()
        torch.cuda.empty_cache()

    print("Saving the latest model...")
    torch.save(student_model.state_dict(), MODEL_SAVE_PATH)
    torch.save(best_model, BEST_MODEL_SAVE_PATH) 


if __name__ == "__main__" : 
    main()
