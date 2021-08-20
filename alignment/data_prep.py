import os
import shutil

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

import __init__  # need this to get back to main project directory (hacky solution, maybe there's a better way)
config = utils.get_config()

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
        if modality == 'speech':
            for file in filelist:
                filename = file.split('/')[-1][:-5]  # remove .flac
                new_name = filename + '_' + modality
                if not args.speaker_dependent:
                    dump_path = os.path.join(args.dump_dir, new_name + '.wav')  # save audio files as .wav for alignment
                else:
                    dump_dir = os.path.join(args.dump_dir, filename)
                    os.makedirs(dump_dir, exist_ok=True)
                    dump_path = os.path.join(dump_dir, new_name + '.wav')  # save audio files as .wav for alignment
                file_rename_list.append({'og_path': file, 'dump_path': dump_path})

    multiproc_data = []
    for file in file_rename_list:
        multiproc_data.append({'file': file})
    # process(multiproc_data[100])
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_task) as executor:
        for _ in tqdm(executor.map(process, multiproc_data)):
            """"""



def process(data):
    """"""
    file = data['file']
    og_path = file['og_path']
    dump_path = file['dump_path']

    audio, _ = librosa.core.load(og_path, sr=16000)  # fix sample rate to 16kHz for kaldi compatibility
    audio = librosa.util.normalize(audio)
    utils.dump_audio(audio, dump_path, sr=16000)

    # now copy the common_transcript.txt file to the same directory with the same name as the audio file
    txt_dump = dump_path[:-4] + '.lab'
    shutil.copy(src='alignment/common_transcript.txt', dst=txt_dump)



def main(args):
    """"""
    if not os.path.isdir(args.dump_dir):
        os.makedirs(args.dump_dir, exist_ok=True)
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.isdir(args.tmp_dir):
        os.makedirs(args.tmp_dir, exist_ok=True)
    if args.dataset == 'DiCOVA':
        prepare_DiCOVA(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments to extract features')
    parser.add_argument('--dataset_root', type=str, default='DiCOVA')  # root of dataset for which to extract features
    parser.add_argument('--dataset', type=str, default='DiCOVA')  # dataset name
    parser.add_argument('--dump_dir', type=str, default='alignment/DiCOVA')  # where to dump .wav files
    parser.add_argument('--speaker_dependent', action='store_true', default=True)
    parser.add_argument('--num_task', type=int, default=8)  # number of cpus for multiprocessing
    parser.add_argument('--out_dir', type=str, default='alignment/DiCOVA_aligned')  # where to dump alignment files
    parser.add_argument('--tmp_dir', type=str, default='alignment/tmp')  # temp directory for alignment
    args = parser.parse_args()
    main(args)