import tensorflow_datasets as tfds
import os
cfg = tfds.download.DownloadConfig(extract_dir=os.getcwd() + '/raw_datasets/')
tfds.load('speech_commands', data_dir='./raw_datasets', download=True, download_and_prepare_kwargs={'download_dir': os.getcwd() + '/raw_datasets/', 'download_config': cfg})
