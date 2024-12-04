import pandas as pd
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pynput.keyboard import Key, Controller, Listener
from eeg_interface.models.threshold import Threshold
from brainflow.data_filter import DataFilter, FilterTypes
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.signal import detrend



class EegStream:
    def __init__(self):
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.serial_port = 'COM3'
        self.markers = []
        self.is_space_pressed = False 
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        self.ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value)
        self.stop_streaming = False
        self.threshold = Threshold(-65830.536583726)
        


    def on_press(self,key):
        global is_space_pressed
        if key == Key.space:
            self.is_space_pressed = True
        if key == Key.esc:
            self.stop_streaming=True


    def on_release(self,key):
        global is_space_pressed
        if key == Key.space:
            self.is_space_pressed = False
            
    def remove_outliers(self,signal, threshold=-1000):
        # Remplace les valeurs hors de la plage [-threshold, threshold] par la moyenne locale
        return signal[signal > threshold]
    
    

    def apply_median_filter(self,signal, kernel_size=5):
        # Applique un filtre médian avec une fenêtre de taille `kernel_size`
        return medfilt(signal, kernel_size)
    
    def Play(self):
        controller = Controller()
        self.board.prepare_session()
        self.board.start_stream()
        sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
        frontal_channel = eeg_channels[1]  # Exemple : premier canal frontal
        min_distance = int(0.2 * sampling_rate)  # Minimum 200 ms entre deux clignements
        num_samples = 20
        try:
            while not self.stop_streaming:  

                data = self.board.get_current_board_data(num_samples)
                if data.shape[1] > 0:  # Vérifiez qu'il y a des données
                    frontal_signal = data[frontal_channel]  # Canal frontal
                    gain = 24  # Valeur typique pour Cyton
                    frontal_signal = frontal_signal * (4.5 * 1e6) / (24 * 8388608)
                    # DataFilter.perform_bandpass(frontal_signal, sampling_rate, 0.5, 10.0, 4, FilterTypes.BUTTERWORTH.value, 0)
                    DataFilter.perform_highpass(frontal_signal, sampling_rate, 0.5, 4, FilterTypes.BUTTERWORTH.value, 0)
                    frontal_signal = self.remove_outliers(frontal_signal, threshold=0)                                                                                    
                    # Filtrage médian pour lisser les artefacts
                    # frontal_signal = medfilt(frontal_signal, kernel_size=5)

                    
                    # Filtrage passe-bande 0.5-10 Hz

                    print(frontal_signal)
                    # Détection des pics
                    threshold = np.mean(frontal_signal) + 3* np.std(frontal_signal)  # Seuil dynamique
                    peaks = np.where(frontal_signal  < 1)[0]  # Indices des pics détectés
                    # Filtrer les pics pour éviter les doublons
                    filtered_peaks = []
                    for peak in peaks:
                        if not filtered_peaks or (peak - filtered_peaks[-1] > min_distance):
                            filtered_peaks.append(peak)

                    # Affichage des clignements détectés
                    if filtered_peaks and not self.is_space_pressed:
                        self.is_space_pressed = True
                        print(f"Clignements détectés : indices = {filtered_peaks}")
                        print('___________')
                        controller.press(Key.space)
                        controller.release(Key.space)
                    elif not filtered_peaks and self.is_space_pressed:
                        self.is_space_pressed = False
                        print('lock open  ')
                    time.sleep(0.001)
                # for row in eeg_data.T:          
                #     if self.threshold.shouldFlap(row[1]) and not self.is_space_pressed:
                #         # controller.press(Key.space)
                #         # controller.release(Key.space)
                #         pass
                #     elif not self.threshold.shouldFlap(row[1]) and self.is_space_pressed:
                #         self.is_space_pressed = False
                    
            
        except KeyboardInterrupt:
            print("Capture interrompue.")
        
            
    def record(self):
        
        print("Streaming des données pendant 5 minutes. Appuyez sur la touche Espace pour ajouter un marqueur.")
        start_time = time.time()
        listener = Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        data_buffer = []
        self.board.prepare_session()
        self.board.start_stream()

        try:
            while time.time() - start_time < 5*60:  

                data = self.board.get_current_board_data(5)
                eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)  # Liste des indices des canaux EEG
                eeg_data = data[eeg_channels, :]
                for row in eeg_data.T:
                    print(row)
                    if self.is_space_pressed:
                        self.markers.append(1)
                    else:
                        self.markers.append(0)
                    data_buffer.append(row)
                    break
                time.sleep(0.1)  

        except KeyboardInterrupt:
            print("Capture interrompue.")

        self.board.stop_stream()
        self.board.release_session()
        columns = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value) + ["Marker"]
        print(columns)
        data_with_markers = [list(row) + [marker] for row, marker in zip(data_buffer, self.markers)]
        print(len(data_with_markers))
        print(len(data_with_markers[0]))
        self.df = pd.DataFrame(data_with_markers, columns=columns)
        print(self.df)
        self.df.to_csv('./Data/EyesClosed/1.csv')
        listener.stop()
        