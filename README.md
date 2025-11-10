decisiontree-and-vnn

CIFAR-10 image classification using:

Decision Tree (NumPy from scratch)

Decision Tree (scikit-learn)

Convolutional Neural Network (VGG11 in PyTorch)

The repo follows the COMP472 assignment spec: DT trained on 50-D features (ResNet-18 features + PCA), and VGG11 trained directly on CIFAR-10 images. All runs save confusion matrices and metrics (accuracy, precision, recall, F1).

Project Structure

.
├── data/                      # CIFAR-10 is auto-downloaded here
├── features/                  # 50-D CSV features saved here
│   ├── train_features.csv
│   └── test_features.csv
├── models/                    # Saved PyTorch model(s)
│   └── vgg11_main.pth
├── results_dt_numpy/          # Outputs from NumPy Decision Tree (confusion matrices + metrics)
├── results_dt_sklearn/        # Outputs from sklearn Decision Tree (confusion matrices + metrics)
├── results_vgg11/             # Outputs from VGG11 (confusion matrix + metrics)
├── decision_tree_numpy.py     # DT (from scratch, NumPy)
├── decision_tree_sklearn.py   # DT (sklearn)
├── vgg11_cnn.py               # VGG11 (PyTorch)
├── make_csv_features.py       # ResNet18 feature extractor + PCA(50) -> CSVs
└── eval_utils.py              # Metrics + confusion matrix plotting

Requirements

Python 3.9+

pip install numpy pandas scikit-learn torch torchvision matplotlib

Apple Silicon (M1/M2/M3):
pip install torch torchvision torchaudio (MPS is used automatically on macOS; you can also use the CPU wheels via the PyTorch download page).

1) Create Feature CSVs (for Decision Trees)

This script:

Downloads CIFAR-10 to data/ (if needed)

Extracts 512-D features with ResNet-18 (pretrained, fc=Identity)

Reduces to 50-D with PCA

Saves to features/train_features.csv and features/test_features.csv

python make_csv_features.py


Expected output:

features/
 ├── train_features.csv  # columns f1..f50 + label
 └── test_features.csv

2) Decision Tree (NumPy – from scratch)

Train and evaluate across multiple depths (example: 5, 10, 20, 50):

python decision_tree_numpy.py \
  --train features/train_features.csv \
  --test  features/test_features.csv \
  --outdir results_dt_numpy \
  --depths 5 10 20 50


Artifacts:

results_dt_numpy/
 ├── cm_depth5.png
 ├── metrics_depth5.json
 ├── cm_depth10.png
 ├── metrics_depth10.json
 └── ... (etc.)

3) Decision Tree (scikit-learn)
python decision_tree_sklearn.py \
  --train features/train_features.csv \
  --test  features/test_features.csv \
  --outdir results_dt_sklearn \
  --depths 5 10 20 50


Artifacts:

results_dt_sklearn/
 ├── cm_depth5.png
 ├── metrics_depth5.json
 ├── cm_depth10.png
 ├── metrics_depth10.json
 └── ... (etc.) + summary.json

4) CNN – VGG11 (PyTorch)

VGG11 is implemented per the assignment (Conv-BN-ReLU-MaxPool blocks + Linear-Dropout head), trained with CrossEntropyLoss and SGD (momentum=0.9) on CIFAR-10 images.

Train
python vgg11_cnn.py \
  --mode train \
  --epochs 30 \
  --batch_size 128 \
  --lr 0.01 \
  --outdir results_vgg11 \
  --kernel_size 3

Evaluate a saved model
python vgg11_cnn.py \
  --mode eval \
  --checkpoint models/vgg11_main.pth \
  --outdir results_vgg11_eval


Artifacts:

models/
 └── vgg11_main.pth           # best checkpoint from training

results_vgg11/
 ├── cm_vgg11_k3.png          # confusion matrix
 └── metrics_vgg11_k3.json    # accuracy, precision, recall, F1


To run kernel-size variants (required): re-run training with --kernel_size 2, 5, and 7, saving each run’s metrics and CM.

📈 Outputs & Where They’re Saved

Confusion matrices (PNG): in results_dt_numpy/, results_dt_sklearn/, results_vgg11/

Metrics (JSON): same folders; include accuracy, macro precision/recall/F1, and per-class stats

Saved models: models/vgg11_main.pth (for later evaluation)

🔁 Reproducibility

Random seeds are set in the code (NumPy & PyTorch).

Re-running scripts will recreate data/, features/, models/, and results_* if missing.

🧹 Cleaning Up

Safe to delete (will be regenerated if you run scripts again):

data/         # CIFAR-10 download
features/     # feature CSVs (recreate via make_csv_features.py)
results_dt_numpy/
results_dt_sklearn/
results_vgg11/


Important: You can delete the models/ folder for a clean repo.
If deleted, your teammate can’t evaluate your trained model unless they retrain it (30 epochs).

🧪 Example One-liners
# 1) Features
python make_csv_features.py

# 2) Decision Tree (NumPy)
python decision_tree_numpy.py --train features/train_features.csv --test features/test_features.csv --outdir results_dt_numpy --depths 5 10 20 50

# 3) Decision Tree (sklearn)
python decision_tree_sklearn.py --train features/train_features.csv --test features/test_features.csv --outdir results_dt_sklearn --depths 5 10 20 50

# 4) VGG11 train/eval
python vgg11_cnn.py --mode train --epochs 30 --batch_size 128 --lr 0.01 --outdir results_vgg11 --kernel_size 3
python vgg11_cnn.py --mode eval --checkpoint models/vgg11_main.pth --outdir results_vgg11_eval

❓Troubleshooting

ModuleNotFoundError: torch
pip install torch torchvision (see PyTorch site for platform-specific wheels)

Features CSV missing
Run python make_csv_features.py to generate features/train_features.csv and features/test_features.csv.

Slow training on macOS
PyTorch will use MPS automatically when available; otherwise it falls back to CPU.