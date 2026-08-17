import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from dataset import get_dataloaders
from model import SpatiotemporalMAE

# My hardcoded labels for the N=11 experiment
# 0 = Normal, 1 = Pathology
patient_ground_truth = {
    'SCD2001_000': 1, 'SCD2001_001': 0, 'SCD2001_002': 1,
    'SCD2001_003': 1, 'SCD2001_004': 0, 'SCD2001_005': 1,
    'SCD2001_006': 0, 'SCD2001_007': 1, 'SCD2001_008': 0,
    'SCD2001_009': 1, 'SCD2001_010': 0 
}

def get_frozen_feats(dataset, p_ids, net, device):
    feat_list, label_list = [], []
    net.eval()
    
    with torch.no_grad():
        for i in range(len(dataset)):
            seq_tensor = dataset[i].unsqueeze(0).to(device)
            current_pid = p_ids[i]
            
            # pass through encoder only!
            raw_feats = net.encoder(seq_tensor.permute(0, 2, 1, 3, 4))
            
            # squash spatial dims down to 1D vector
            squashed = nn.AdaptiveAvgPool3d((1, 1, 1))(raw_feats).flatten()
            
            feat_list.append(squashed.cpu().numpy())
            label_list.append(patient_ground_truth[current_pid])
            
    return np.vstack(feat_list), np.array(label_list)

def test_svm_downstream():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, train_ds, val_ds, train_pids, val_pids = get_dataloaders(batch_size=1)
    
    mae_net = SpatiotemporalMAE(mask_ratio=0.75).to(device)
    
    if os.path.exists("mae_frozen_encoder.pth"):
        mae_net.load_state_dict(torch.load("mae_frozen_encoder.pth", map_location=device))
        print("Loaded encoder weights for feature extraction.")
    
    print("Extracting features (this might take a second)...")
    X_train, y_train = get_frozen_feats(train_ds, train_pids, mae_net, device)
    X_val, y_val = get_frozen_feats(val_ds, val_pids, mae_net, device)
    
    print("Fitting SVM...")
    # tried rbf kernel first but it overfit immediately, linear is better here
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    svm_preds = clf.predict(X_val)
    svm_probs = clf.predict_proba(X_val)[:, 1]
    
    final_acc = accuracy_score(y_val, svm_preds)
    final_auc = roc_auc_score(y_val, svm_probs)
    
    print(f"\n--- PROBE RESULTS ---")
    print(f"Accuracy: {final_acc * 100:.2f}%")
    print(f"AUC Score: {final_auc:.4f}")
    print("As expected, it drops to random chance (50%) on unseen patients.")

if __name__ == "__main__":
    test_svm_downstream()