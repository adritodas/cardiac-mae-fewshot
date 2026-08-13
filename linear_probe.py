import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from dataset import get_dataloaders
from model import SpatiotemporalMAE

# Hardcoded labels for N=11 simulation (0=Normal, 1=Pathology)
patient_labels = {
    'SCD2001_000': 1, 'SCD2001_001': 0, 'SCD2001_002': 1,
    'SCD2001_003': 1, 'SCD2001_004': 0, 'SCD2001_005': 1,
    'SCD2001_006': 0, 'SCD2001_007': 1, 'SCD2001_008': 0,
    'SCD2001_009': 1, 'SCD2001_010': 0 
}

def extract_features(dataset, patient_ids, model, device):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            seq = dataset[i].unsqueeze(0).to(device)
            p_id = patient_ids[i]
            
            feat = model.encoder(seq.permute(0, 2, 1, 3, 4))
            feat_vector = nn.AdaptiveAvgPool3d((1, 1, 1))(feat).flatten()
            
            features.append(feat_vector.cpu().numpy())
            labels.append(patient_labels[p_id])
            
    return np.vstack(features), np.array(labels)

def run_svm_probe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, train_dataset, val_dataset, train_patients, val_patients = get_dataloaders(batch_size=1)
    
    model = SpatiotemporalMAE(mask_ratio=0.75).to(device)
    
    if os.path.exists("mae_frozen_encoder.pth"):
        model.load_state_dict(torch.load("mae_frozen_encoder.pth", map_location=device))
        print("Loaded frozen MAE encoder weights.")
    else:
        print("Warning: No weights found. Running with initialized weights.")
    
    print("Extracting frozen spatiotemporal features...")
    X_train, y_train = extract_features(train_dataset, train_patients, model, device)
    X_val, y_val = extract_features(val_dataset, val_patients, model, device)
    
    print("Training SVM Classifier...")
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    preds_binary = clf.predict(X_val)
    preds_proba = clf.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, preds_binary)
    auc = roc_auc_score(y_val, preds_proba)
    
    print(f"\n--- FINAL MAE PROBE METRICS ---")
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    print(f"Validation AUC:      {auc:.4f}")

if __name__ == "__main__":
    run_svm_probe()