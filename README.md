# COVID-19 Chest X-Ray Classifier

Projet IPSA AERO4 — Détection COVID-19, Pneumonie et Normal sur radios pulmonaires (Blanc/Noir)  
**Modèles utilisés :** CNN (TensorFlow/Keras), SVM & MLP (scikit-learn)

# Covid-detector
Ce projet développe un système d'aide au diagnostic par IA pour la détection automatisée de la COVID-19 à partir de radiographies pulmonaires. Face à la surcharge des services de radiologie, ce prototype vise à servir d'outil de premier tri, capable de distinguer les cas de COVID-19 des pneumonies virales et des poumons sains.

---

## Description

Ce projet propose une pipeline complète pour la classification automatique de radios thoraciques en trois classes :
- **COVID-19**
- **Pneumonia**
- **Normal**

Deux approches sont comparées :
- **CNN** (Convolutional Neural Network) avec TensorFlow/Keras
- **SVM** (Support Vector Machine) avec scikit-learn

---

## Structure du projet

```
covid_detector/
│
├── data/                # Dossier contenant les images classées par dossier (covid, pneumonia, normal)
├── script/
│   ├── Cnn_model_training.py      # Entraînement et évaluation du CNN
│   └── Svm_model_training.py       # Entraînement et évaluation SVM
├── eda_covid19_bn.png   # Visualisation EDA (pie chart, radios exemples)
├── resultat2.png        # Résultats CNN (courbes, matrice de confusion)
└── covid19_bn_cnn_final.pkl   # Modèle CNN sauvegardé (50 epochs)
```

---

## Installation

1. **Cloner le dépôt**
   ```bash
   git clone <url_du_repo>
   cd covid_detector
   ```

2. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
   Ou manuellement :
   ```bash
   pip install numpy matplotlib seaborn scikit-learn pillow tensorflow joblib tqdm
   ```

3. **Préparer les données**
   - Placez vos images dans `data/` avec un sous-dossier par classe (`covid`, `pneumonia`, `normal`).

---

## Utilisation

### 1. Entraînement CNN

```bash
python3 script/Cnn_model_training.py (mac)
```
- Génère les courbes d'apprentissage, matrices de confusion, et sauvegarde le modèle.

### 2. Entraînement SVM

```bash
python3 script/Svm_model_training.py
```
- Génère les matrices de confusion et sauvegarde le modèle SVM.

### 3. Exécution simultanée (optionnel)

Dans deux terminaux :
```bash
python3 script/Cnn_model_training.py
python3 script/Svm_model_training.py
```
Ou en arrière-plan :
```bash
nohup python3 script/Cnn_model_training.py > cnn.log 2>&1 &
nohup python3 script/Svm_model_training.py > svm.log 2>&1 &
```

---

## Résultats

- **Courbes d'apprentissage** : accuracy et loss (train/validation)
- **Matrices de confusion** : pour chaque modèle
- **Rapport de classification** : précision, rappel, f1-score par classe

---

## GPU

TensorFlow utilisera le GPU automatiquement si disponible (NVIDIA + CUDA/cuDNN).  
Sur Mac M1/M2, privilégier Google Colab pour l'accélération matérielle.

---

## Auteurs

- Killian (IPSA AERO4)
- Projet pédagogique, 2025

---

## Licence

Projet académique — usage pédagogique uniquement.
