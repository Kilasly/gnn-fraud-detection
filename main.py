# ===============================================================
# ELLIPTIC BITCOIN - GNN FRAUD DETECTION SYSTEM
# GRAPH CONVOLUTIONAL NETWORK (GCN)
# ===============================================================

# ---------------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------

print("\nLoading dataset...")

features = pd.read_csv(
    "elliptic_txs_features.csv",
    header=None
)

edges = pd.read_csv(
    "elliptic_txs_edgelist.csv"
)

labels = pd.read_csv(
    "elliptic_txs_classes.csv"
)

print("Dataset loaded successfully.")

print("\nDataset Shapes:")
print("Features:", features.shape)
print("Edges:", edges.shape)
print("Labels:", labels.shape)


# ---------------------------------------------------------------
# 2. REMAP NODE IDS
# ---------------------------------------------------------------

print("\nRemapping transaction IDs...")

# Original transaction IDs
tx_ids = features.iloc[:, 0].values

# Create mapping dictionary
id_map = {
    original_id: new_id
    for new_id, original_id in enumerate(tx_ids)
}

# Copy edge list
edges_remapped = edges.copy()

# Replace original IDs
edges_remapped["txId1"] = edges["txId1"].map(id_map)
edges_remapped["txId2"] = edges["txId2"].map(id_map)

# Remove invalid rows
edges_remapped = edges_remapped.dropna().astype(int)

print("Remapping completed.")
print("Edges shape:", edges_remapped.shape)


# ---------------------------------------------------------------
# 3. PREPARE NODE FEATURES
# ---------------------------------------------------------------

print("\nPreparing node features...")

# Remove txId and timestep columns
feature_values = features.iloc[:, 2:].values

# Normalize features
scaler = StandardScaler()

feature_values = scaler.fit_transform(
    feature_values
)

# Convert to tensor
x = torch.tensor(
    feature_values,
    dtype=torch.float
)

print("Feature tensor shape:", x.shape)


# ---------------------------------------------------------------
# 4. PREPARE LABELS
# ---------------------------------------------------------------

print("\nPreparing labels...")

# Convert labels to string
labels["class"] = labels["class"].astype(str)

print("\nUnique labels:")
print(labels["class"].unique())

# Label mapping
# 1 = illicit/fraud
# 2 = licit/normal
# unknown = unlabeled

def map_label(value):

    if value == "unknown":
        return -1

    elif value == "1":
        return 1

    elif value == "2":
        return 0

    else:
        return -1


# Apply mapping
labels_mapped = labels["class"].apply(
    map_label
)

# Convert to tensor
y = torch.tensor(
    labels_mapped.values,
    dtype=torch.long
)

print("\nLabel Distribution:")
print(labels_mapped.value_counts())

print("\nLabel tensor shape:")
print(y.shape)


# ---------------------------------------------------------------
# 5. BUILD GRAPH DATA OBJECT
# ---------------------------------------------------------------

print("\nBuilding graph object...")

# Create edge index
edge_index = torch.tensor(
    edges_remapped.values.T,
    dtype=torch.long
)

# Create graph object
data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

print("\nGraph created successfully.")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of features:", data.num_node_features)


# ---------------------------------------------------------------
# 6. CREATE TRAIN / VALIDATION / TEST SPLITS
# ---------------------------------------------------------------

print("\nCreating train/validation/test splits...")

# Use only labeled nodes
labeled_nodes = np.where(
    (data.y == 0) | (data.y == 1)
)[0]

# Shuffle nodes
np.random.shuffle(labeled_nodes)

# Define split sizes
train_size = int(0.6 * len(labeled_nodes))
val_size = int(0.2 * len(labeled_nodes))

# Split indices
train_idx = labeled_nodes[:train_size]

val_idx = labeled_nodes[
    train_size:train_size + val_size
]

test_idx = labeled_nodes[
    train_size + val_size:
]

# Create masks
train_mask = torch.zeros(
    data.num_nodes,
    dtype=torch.bool
)

val_mask = torch.zeros(
    data.num_nodes,
    dtype=torch.bool
)

test_mask = torch.zeros(
    data.num_nodes,
    dtype=torch.bool
)

# Assign masks
train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

print("Train nodes:", train_mask.sum().item())
print("Validation nodes:", val_mask.sum().item())
print("Test nodes:", test_mask.sum().item())


# ---------------------------------------------------------------
# 7. DEFINE GCN MODEL
# ---------------------------------------------------------------

print("\nBuilding GCN model...")


class GCN(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim
    ):

        super().__init__()

        # First GCN layer
        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        # Second GCN layer
        self.conv2 = GCNConv(
            hidden_dim,
            output_dim
        )

    def forward(
        self,
        x,
        edge_index
    ):

        # Layer 1
        x = self.conv1(
            x,
            edge_index
        )

        # Activation
        x = F.relu(x)

        # Dropout
        x = F.dropout(
            x,
            p=0.5,
            training=self.training
        )

        # Layer 2
        x = self.conv2(
            x,
            edge_index
        )

        return x


# Create model
model = GCN(
    input_dim=data.num_node_features,
    hidden_dim=64,
    output_dim=2
)

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)

print("GCN model ready.")


# ---------------------------------------------------------------
# 8. TRAINING FUNCTION
# ---------------------------------------------------------------

# Handle class imbalance
class_weights = torch.tensor(
    [1.0, 5.0]
)

def train_step():

    model.train()

    optimizer.zero_grad()

    # Forward pass
    out = model(
        data.x,
        data.edge_index
    )

    # Cross entropy loss
    loss = F.cross_entropy(
        out[train_mask],
        data.y[train_mask],
        weight=class_weights
    )

    # Backpropagation
    loss.backward()

    optimizer.step()

    return loss.item()


# ---------------------------------------------------------------
# 9. EVALUATION FUNCTION
# ---------------------------------------------------------------

@torch.no_grad()
def evaluate(mask):

    model.eval()

    out = model(
        data.x,
        data.edge_index
    )

    predictions = out.argmax(dim=1)

    correct = (
        predictions[mask] ==
        data.y[mask]
    ).sum().item()

    total = mask.sum().item()

    if total == 0:
        return 0

    accuracy = correct / total

    return accuracy


# ---------------------------------------------------------------
# 10. TRAIN MODEL
# ---------------------------------------------------------------

print("\nTraining model...")

best_val_accuracy = 0

patience = 20
wait = 0

# Store metrics
train_losses = []
val_accuracies = []

for epoch in range(1, 301):

    # Train
    loss = train_step()

    # Validate
    val_accuracy = evaluate(val_mask)

    # Save metrics
    train_losses.append(loss)
    val_accuracies.append(val_accuracy)

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        wait = 0

        torch.save(
            model.state_dict(),
            "best_gcn_model.pt"
        )

    else:
        wait += 1

    # Print progress
    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

    # Early stopping
    if wait >= patience:

        print("\nEarly stopping triggered.")

        break


# ---------------------------------------------------------------
# 11. VISUALIZE TRAINING
# ---------------------------------------------------------------

print("\nGenerating training graphs...")

plt.figure(figsize=(12, 5))

# Training loss graph
plt.subplot(1, 2, 1)

plt.plot(train_losses)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Validation accuracy graph
plt.subplot(1, 2, 2)

plt.plot(val_accuracies)

plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()

plt.show()


# ---------------------------------------------------------------
# 12. LOAD BEST MODEL
# ---------------------------------------------------------------

print("\nLoading best model...")

model.load_state_dict(
    torch.load("best_gcn_model.pt")
)

test_accuracy = evaluate(test_mask)

print("\nFINAL TEST ACCURACY:")
print(round(test_accuracy, 4))


# ---------------------------------------------------------------
# 13. CLASSIFICATION REPORT
# ---------------------------------------------------------------

print("\nGenerating classification report...")

model.eval()

out = model(
    data.x,
    data.edge_index
)

predictions = out.argmax(dim=1)

# True labels
y_true = data.y[test_mask].cpu().numpy()

# Predicted labels
y_pred = predictions[test_mask].cpu().numpy()

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred
    )
)


# ---------------------------------------------------------------
# 14. CONFUSION MATRIX
# ---------------------------------------------------------------

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:\n")
print(cm)


# ---------------------------------------------------------------
# 15. ANOMALY DETECTION
# ---------------------------------------------------------------

print("\nRunning anomaly detection...")

# Fraud probabilities
probabilities = F.softmax(
    out,
    dim=1
)[:, 1]

# Fraud predictions
fraud_nodes = torch.where(
    predictions == 1
)[0]

print(
    "Number of predicted fraudulent transactions:",
    len(fraud_nodes)
)


# ---------------------------------------------------------------
# 16. SAVE RESULTS
# ---------------------------------------------------------------

print("\nSaving prediction results...")

results = pd.DataFrame({

    "txId": features.iloc[:, 0].values,

    "true_label": data.y.numpy(),

    "predicted_class": predictions.numpy(),

    "fraud_probability": probabilities.detach().numpy()
})

results.to_csv(
    "elliptic_gnn_predictions.csv",
    index=False
)

print("Results saved successfully.")

print("\nSYSTEM COMPLETED SUCCESSFULLY ✔")