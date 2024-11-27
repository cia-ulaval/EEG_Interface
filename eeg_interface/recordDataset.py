from eeg_interface.data_handler.eegStream import EegStream


def cli():
    stream = EegStream()
    for i in range(1):
        stream.record()

if __name__ == '__main__':
    cli()