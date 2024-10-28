import numpy as np
import pandas as pd
from torcheeg import transforms

# Charger les données EEG depuis un fichier CSV
df = pd.read_csv('chemin/vers/votre_fichier.csv')
eeg = df.values  # Convertir le DataFrame en tableau NumPy


# TRANFORMATION D'ENTROPIE DIFFÉRENTIELLE DE BANDE
# Appliquer la transformation BandDifferentialEntropy
transformed_eeg_entropy = transforms.BandDifferentialEntropy()(eeg=eeg)['eeg']
# Afficher ou utiliser les données EEG transformées par BandDifferentialEntropy
print("Transformed EEG with BandDifferentialEntropy:")
print(transformed_eeg_entropy)


# ANALYSE FRACTALE DE HIGUCHI
# Appliquer la transformation BandHiguchiFractalDimension
transformed_eeg_higuchi = transforms.BandHiguchiFractalDimension()(eeg=eeg)['eeg']
# Afficher ou utiliser les données EEG transformées par BandHiguchiFractalDimension
print("Transformed EEG with BandHiguchiFractalDimension:")
print(transformed_eeg_higuchi)


# DENSITÉ SPECTRALE DE PUISSANCE DE BANDE
# Appliquer la transformation BandPowerSpectralDensity
transformed_eeg_psd = transforms.BandPowerSpectralDensity()(eeg=eeg)['eeg']
# Afficher ou utiliser les données EEG transformées par BandPowerSpectralDensity
print("Transformed EEG with BandPowerSpectralDensity:")
print(transformed_eeg_psd)
