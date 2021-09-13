import os
from tqdm import tqdm
import numpy as np
import joblib
from utils import collect_files
import utils
from torch.utils import data
import json
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict
import argparse
import librosa
import scipy.signal as Signal
import random
import multiprocessing
import concurrent.futures
from pydub import AudioSegment

config = utils.get_config()


def webm_to_wav(data):
    file = data['file']
    tmp_dump_dir = data['tmp_dump_dir']
    try:
        filename = file.split('/')[-1][:-5]  # remove .webm
        dump_path = os.path.join(tmp_dump_dir, filename + '.wav')
        song = AudioSegment.from_file(file, "webm")
        song.export(dump_path, format="wav")
    except:
        print("File failed...")

def prepare_DiCOVA(args):
    """Collect all the audio files from their respective folders and provide
       dict with original filepath and write name"""

    files = {}
    # Breathing #
    files['breathing'] = utils.keep_flac(utils.collect_files(os.path.join(args.dataset_root, 'AUDIO', 'breathing')))
    # Cough #
    files['cough'] = utils.keep_flac(utils.collect_files(os.path.join(args.dataset_root, 'AUDIO', 'cough')))
    # Speech #
    files['speech'] = utils.keep_flac(utils.collect_files(os.path.join(args.dataset_root, 'AUDIO', 'speech')))

    file_rename_list = []
    for modality, filelist in files.items():
        for file in filelist:
            filename = file.split('/')[-1][:-5]  # remove .flac
            new_name = filename + '_' + modality
            dump_path = os.path.join(args.dump_dir, new_name + '.pkl')
            file_rename_list.append({'og_path': file, 'dump_path': dump_path})

    mel_log_spect = utils.Mel_log_spect()

    multiproc_data = []
    for file in file_rename_list:
        multiproc_data.append({'file': file, 'feat_extractor': mel_log_spect})
    # process(multiproc_data[2100])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_task) as executor:
        for _ in tqdm(executor.map(process, multiproc_data)):
            """"""

def prepare_DiCOVA_Test(args):
    """Collect all the audio files from their respective folders and provide
       dict with original filepath and write name"""

    files = {}
    # Breathing #
    files['breathing'] = utils.keep_flac(utils.collect_files(os.path.join(args.dataset_root, 'AUDIO', 'breathing')))
    # Cough #
    files['cough'] = utils.keep_flac(utils.collect_files(os.path.join(args.dataset_root, 'AUDIO', 'cough')))
    # Speech #
    files['speech'] = utils.keep_flac(utils.collect_files(os.path.join(args.dataset_root, 'AUDIO', 'speech')))

    file_rename_list = []
    for modality, filelist in files.items():
        for file in filelist:
            filename = file.split('/')[-1][:-5]  # remove .flac
            new_name = filename + '_' + modality
            dump_path = os.path.join(args.dump_dir, new_name + '.pkl')
            file_rename_list.append({'og_path': file, 'dump_path': dump_path})

    mel_log_spect = utils.Mel_log_spect()

    multiproc_data = []
    for file in file_rename_list:
        multiproc_data.append({'file': file, 'feat_extractor': mel_log_spect})
    # process(multiproc_data[2100])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_task) as executor:
        for _ in tqdm(executor.map(process, multiproc_data)):
            """"""

def prepare_LibriSpeech(args):
    """We use the train-clean-100 partition just as in Interspeech paper"""
    filelist = utils.keep_flac(utils.collect_files(os.path.join(args.dataset_root, 'train-clean-100')))
    file_rename_list = []
    for file in filelist:
        filename = file.split('/')[-1][:-5]  # remove .flac
        new_name = filename
        dump_path = os.path.join(args.dump_dir, new_name + '.pkl')
        file_rename_list.append({'og_path': file, 'dump_path': dump_path})

    mel_log_spect = utils.Mel_log_spect()

    multiproc_data = []
    for file in file_rename_list:
        multiproc_data.append({'file': file, 'feat_extractor': mel_log_spect})
    # process(multiproc_data[2100])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_task) as executor:
        for _ in tqdm(executor.map(process, multiproc_data)):
            """"""

def prepare_COUGHVID(args):
    """We use the train-clean-100 partition just as in Interspeech paper"""
    filelist = utils.keep_webm(utils.collect_files(args.dataset_root))
    """Dump all .webm files to disk as .wav files first"""
    tmp_dump_dir = 'tmp_webm_wav'
    if not os.path.isdir(tmp_dump_dir):
        os.mkdir(tmp_dump_dir)
    multiproc_data = []
    for file in filelist:
        multiproc_data.append({'file': file, 'tmp_dump_dir': tmp_dump_dir})
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_task) as executor:
        for _ in tqdm(executor.map(webm_to_wav, multiproc_data)):
            """"""

    filelist = utils.keep_wavs(utils.collect_files(tmp_dump_dir))
    file_rename_list = []
    for file in filelist:
        filename = file.split('/')[-1][:-4]  # remove .wav
        new_name = filename
        dump_path = os.path.join(args.dump_dir, new_name + '.pkl')
        file_rename_list.append({'og_path': file, 'dump_path': dump_path})

    mel_log_spect = utils.Mel_log_spect()

    multiproc_data = []
    for file in file_rename_list:
        multiproc_data.append({'file': file, 'feat_extractor': mel_log_spect})
    # process(multiproc_data[80])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_task) as executor:
        for _ in tqdm(executor.map(process, multiproc_data)):
            """"""



def process(data):
    """"""
    file = data['file']
    og_path = file['og_path']
    dump_path = file['dump_path']
    feat_extractor = data['feat_extractor']

    audio, _ = librosa.core.load(og_path, sr=config.data.sr)
    audio = librosa.util.normalize(audio)
    features = feat_extractor.get_Mel_log_spect(audio)

    # plt.imshow(features.T)
    # plt.show()
    """Now we have the features, so we save to disk"""
    utils.dump(features, dump_path, type='pickle')



def main(args):
    """"""
    if not os.path.isdir(args.dump_dir):
        os.makedirs(args.dump_dir, exist_ok=True)
    if args.dataset == 'DiCOVA':
        prepare_DiCOVA(args)
    if args.dataset == 'DiCOVA_Test':
        prepare_DiCOVA_Test(args)
    elif args.dataset == 'COUGHVID':
        """"""
        prepare_COUGHVID(args)
    elif args.dataset == 'LibriSpeech':
        """"""
        prepare_LibriSpeech(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to extract features')
    parser.add_argument('--dataset_root', type=str, default='DiCOVA_Test')  # root of dataset for which to extract features
    parser.add_argument('--dataset', type=str, default='DiCOVA_Test')  # dataset name
    parser.add_argument('--dump_dir', type=str, default='feats/DiCOVA_Test')  # where to dump all spectrograms
    parser.add_argument('--num_task', type=int, default=10)  # number of cpus for multiprocessing
    args = parser.parse_args()
    main(args)