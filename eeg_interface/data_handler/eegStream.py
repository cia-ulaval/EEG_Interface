import pandas as pd
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from pynput import keyboard



class EegStream:
    def __init__(self):
        BoardShim.enable_dev_board_logger()

        params = BrainFlowInputParams()
        params.serial_port = 'COM3'
        self.markers = []
        self.is_space_pressed = False 
        self.board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        self.ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value)


    def on_press(self,key):
        global is_space_pressed
        if key == keyboard.Key.space:
            self.is_space_pressed = True


    def on_release(self,key):
        global is_space_pressed
        if key == keyboard.Key.space:
            self.is_space_pressed = False
            
    def record(self):
        
        print("Streaming des données pendant 5 minutes. Appuyez sur la touche Espace pour ajouter un marqueur.")
        start_time = time.time()
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        data_buffer = []
        self.board.prepare_session()
        self.board.start_stream()

        try:
            while time.time() - start_time < 5*60:  

                data = self.board.get_board_data()
                eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)  # Liste des indices des canaux EEG
                eeg_data = data[eeg_channels, :]
                for row in eeg_data.T:
                    if self.is_space_pressed:
                        self.markers.append(1)
                    else:
                        self.markers.append(0)
                    data_buffer.append(row)
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
        