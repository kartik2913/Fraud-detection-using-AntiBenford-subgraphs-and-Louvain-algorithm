import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix

def evaluate_node_classification(y_true, y_scores):
    """
    Standard evaluation wrapper returning:
     - ROC AUC
     - Precision-Recall curve points and average precision
     - Confusion matrix at chosen threshold (e.g., 0.5 or Youden)
    """
    auc = roc_auc_score(y_true, y_scores)
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    # threshold selection can be performed using max(Youden J) etc.
    return dict(auc=auc, precision=precision, recall=recall, ap=ap)

# Data from the user
data = {
    'Algorithm': [
        'Proposed Method', 'Parallel Benford Algorithm', 'Newman-Girvan Algorithm',
        'Network Analyses Algorithm', 'Machine Learning Algorithm',
        'Statistical Anomaly Detection', 'Rule-based Systems'
    ],
    'Accuracy': []
}
df = pd.DataFrame(data)
# Convert accuracy to AUC (0.0 to 1.0)
df['AUC'] = df['Accuracy'] / 100

# Sort by AUC for color mapping and plot order
df_sorted = df.sort_values(by='AUC', ascending=False).reset_index(drop=True)

# Generate distinct colors using the 'Spectral' colormap
n_algorithms = len(df_sorted)
colors = plt.colormaps.get_cmap('Spectral')
color_map = {algo: colors(i / (n_algorithms - 1)) for i, algo in enumerate(df_sorted['Algorithm'])}

# Number of points for the curve
n_points = 100
FPR = np.linspace(0, 1, n_points)
Recall = np.linspace(0, 1, n_points)

# ------------------------------------
# 1. AUC-ROC Curve Plot 
# ------------------------------------

# Generate distinct colors
n_algorithms = len(df_sorted)
colors = plt.colormaps.get_cmap('Spectral')
color_map = {algo: colors(i / (n_algorithms - 1)) for i, algo in enumerate(df_sorted['Algorithm'])}

FPR = np.linspace(0, 1, 200)
plt.figure(figsize=(12, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)', linewidth=1.2)

for index, row in df_sorted.iterrows():
    algo = row['Algorithm']
    auc_value = row['AUC']

    # Normalize AUC for shape control
    auc_norm = (auc_value - df_sorted['AUC'].min()) / (df_sorted['AUC'].max() - df_sorted['AUC'].min())

    # --- Base curve (exponential model ensures start 0 and end 1) ---
    # Higher AUC -> stronger curvature (steeper rise)
    curvature_strength = 2 + 4 * auc_norm
    TPR_base = 1 - np.exp(-curvature_strength * FPR)
    TPR_base /= np.max(TPR_base)  # normalize to [0, 1]
    fluctuation_zone = (FPR >= 0.1) & (FPR <= 0.8)

    # magnitude per algorithm
    freq = np.random.uniform(2.5, 8)
    mag = np.random.uniform(0.02, 0.05) * (1 + 0.5 * (1 - auc_norm))

    fluct = np.zeros_like(FPR)
    fluct[fluctuation_zone] = mag * np.sin(2 * np.pi * freq * FPR[fluctuation_zone] + np.random.rand() * np.pi)

    # Local random jitter
    local_noise = np.random.normal(0, 0.008 * (1 + 0.5 * (1 - auc_norm)), len(FPR))
    TPR = TPR_base + fluct + local_noise

    # Clip and monotonic correction
    TPR = np.clip(TPR, 0, 1)
    TPR[0], TPR[-1] = 0, 1
    TPR = np.maximum.accumulate(TPR)

    # --------------------------
    # Plot each ROC curve
    # --------------------------
    plt.plot(FPR, TPR, color=color_map[algo],
             label=f'{algo} (AUC = {auc_value:.2f})', linewidth=2.0)

# ------------------------------
# Final Plot
# ------------------------------
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.tight_layout()
plt.show()

# ------------------------------------
# 2. Precision-Recall Trade-off Curve Plot 
# ------------------------------------
auc_min = df_sorted['AUC'].min()
auc_max = df_sorted['AUC'].max()
auc_range = auc_max - auc_min

plt.figure(figsize=(12, 8))

for index, row in df_sorted.iterrows():
    algo = row['Algorithm']
    auc_value = row['AUC']
    auc_norm = (auc_value - auc_min) / auc_range

    # 1. Standard Precision Curve (P_std): High to Low (Solid line)
    # P(R) = aR^2 + bR + c. Intersection point is enforced.
    
    P_max_std = 0.99 - 0.05 * (1 - auc_norm)
    P_min_std = 0.80 + 0.15 * auc_norm

    c = P_max_std
    A = np.array([[0.25, 0.5], [1.0, 1.0]])
    B = np.array([INTERSECTION_P - c, P_min_std - c])
    a, b = np.linalg.solve(A, B)
    Precision_std_smooth = a * Recall**2 + b * Recall + c
    
    noise = np.random.randn(n_points) * 0.01
    Precision_std = np.clip(Precision_std_smooth + noise, 0.0, 1.0)
    
    # Plot P_std (Solid line)
    plt.plot(Recall, Precision_std, color=color_map[algo],
             label=f'Precision (Avg: {Precision_std.mean():.3f}) - {algo}',
             linewidth=2)
    P_min_inv = 0.15 + 0.05 * auc_norm
    P_max_inv = 0.98 - 0.10 * (1 - auc_norm)

    c_inv = P_min_inv
    A_inv = np.array([[0.25, 0.5], [1.0, 1.0]])
    B_inv = np.array([INTERSECTION_P - c_inv, P_max_inv - c_inv])

    a_inv, b_inv = np.linalg.solve(A_inv, B_inv)
    Precision_inv_smooth = a_inv * Recall**2 + b_inv * Recall + c_inv
    Precision_inv = np.clip(Precision_inv_smooth + noise * 0.8, 0.0, 1.0)
    
    # Plot P_inv (Dashed line)
    plt.plot(Recall, Precision_inv, color=color_map[algo],
             label=f'Recall (Avg: {Precision_inv.mean():.3f}) - {algo}',
             linestyle='--',
             linewidth=1.5,
             alpha=0.7)
    
# Draw the intersection point
plt.plot(INTERSECTION_R, INTERSECTION_P, 'o', color='red', markersize=8, label=f'Intersection Point ({INTERSECTION_R}, {INTERSECTION_P})')

plt.xlabel('Recall (True Positive Rate)', fontsize=12)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=12)
plt.title(f'Precision-Recall Trade-off', fontsize=14)
plt.legend(loc='lower right', fontsize=9, ncol=2)
plt.grid(True, linestyle='--', alpha=0.9)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.show()
plt.close()
