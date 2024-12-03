from eeg_interface.data_handler.eegStream import EegStream


def cli():
    stream = EegStream()
    stream.Play()

if __name__ == '__main__':
    cli()