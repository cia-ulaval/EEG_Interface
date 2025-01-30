import numpy as np
import pandas as pd
from torcheeg import transforms

class EEGAnalysis:
    def __init__(self, file_path):
        self.df = self.load_data(file_path)
        if self.df is not None:
            self.eeg = self.df.values  # Convertir le DataFrame en tableau NumPy
        else:
            self.eeg = None

    def load_data(self, file_path):
        """Charger les données EEG depuis un fichier CSV."""
        try:
            df = pd.read_csv(file_path)
            print("Données chargées avec succès. Voici les premières lignes :")
            print(df.head())  # Afficher les premières lignes du DataFrame pour vérification
            return df
        except Exception as e:
            print(f"Erreur lors du chargement des données : {e}")
            return None

    def compute_band_differential_entropy(self):
        """Appliquer la transformation BandDifferentialEntropy."""
        if self.eeg is not None:
            try:
                transformed_eeg_entropy = transforms.BandDifferentialEntropy()(eeg=self.eeg)['eeg']
                print("Transformed EEG with BandDifferentialEntropy:")
                print(transformed_eeg_entropy)
                return transformed_eeg_entropy
            except Exception as e:
                print(f"Erreur lors de la transformation BandDifferentialEntropy : {e}")
                return None
        else:
            print("Les données EEG ne sont pas disponibles.")
            return None

    def compute_higuchi_fractal_dimension(self):
        """Appliquer la transformation BandHiguchiFractalDimension."""
        if self.eeg is not None:
            try:
                transformed_eeg_higuchi = transforms.BandHiguchiFractalDimension()(eeg=self.eeg)['eeg']
                print("Transformed EEG with BandHiguchiFractalDimension:")
                print(transformed_eeg_higuchi)
                return transformed_eeg_higuchi
            except Exception as e:
                print(f"Erreur lors de la transformation BandHiguchiFractalDimension : {e}")
                return None
        else:
            print("Les données EEG ne sont pas disponibles.")
            return None

    def compute_band_power_spectral_density(self):
        """Appliquer la transformation BandPowerSpectralDensity."""
        if self.eeg is not None:
            try:
                transformed_eeg_psd = transforms.BandPowerSpectralDensity()(eeg=self.eeg)['eeg']
                print("Transformed EEG with BandPowerSpectralDensity:")
                print(transformed_eeg_psd)
                return transformed_eeg_psd
            except Exception as e:
                print(f"Erreur lors de la transformation BandPowerSpectralDensity : {e}")
                return None
        else:
            print("Les données EEG ne sont pas disponibles.")
            return None
