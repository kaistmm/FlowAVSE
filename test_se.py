import numpy as np
import glob

from tensorboard import summary
import tqdm
import torch
import os
from argparse import ArgumentParser
import time
from pypapi import events, papi_high as high

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module_vi import SpecsDataModule
from sgmse.model import StochasticRegenerationModel
from sgmse.util.other import *

from sgmse.util.other import si_sdr, pad_spec
from pesq import pesq
from pystoi import stoi


import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import librosa
import soundfile

EPS_LOG = 1e-10


def videocap(path, start_frame, for_sync=False): # for VoxCeleb2
    vid_start = int(start_frame//16000*25)
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        frames=[]
        for i in range(vid_start+51):
            ret, img = cap.read()
            if i<vid_start:
                continue

            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (112,112))
                frames.append(img)
            else:
                frames = np.array(frames)
                frames = np.pad(frames, ((0, 51-i), (0,0), (0,0)), 'wrap')
                assert frames.shape == (51, 112, 112), "padding is set wrong"
                return frames
        frames = np.array(frames)
        return frames # (51, H, W)
    else:
        return None



def load_audio_vox(file_path, max_len, sample_rate=16000):
    audio, sample_rate = librosa.load(file_path, sr=sample_rate) # mono as default
    audiosize = audio.shape[0]
    if audiosize < max_len:
        start_frame=0
        shortage = max_len - audiosize
        min_len = sample_rate*30//25 # 최소 30프레임
        if audiosize < min_len:
            return None, 0
        audio = np.pad(audio, (0, shortage), 'wrap')
        if np.all((audio==0)):
            return None, 0
        
    else:
        start_frame = 0
        audio = audio[0:max_len]
        while np.all((audio==0)):
            start_frame += max_len
            if audiosize < start_frame+max_len:
                return None, 0
            audio = audio[start_frame:start_frame+max_len]
        
    
    return audio, start_frame

def activelev(data):
    try:
        max_amp = np.std(data)
    except:
        return data
    return data/max_amp

def main():
    # Tags
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument("--ckpt", type=str, default='/mnt/bear2/users/cyong/log_FM/mode=regen-joint-training_sde=OUVESDE_score=ncsnpp_crossatt_data=voxceleb2_SE_ch=1/version_66/checkpoints/last.ckpt')
        parser_.add_argument("--mode", type=str, default="storm", choices=["score-only", "denoiser-only", "storm"])
        parser_.add_argument('--log_path', type=str, default='./test.txt')
        parser_.add_argument("--testset", default='lrs3', type=str, choices=['lrs3', 'vox'])
        parser_.add_argument("--noise_path", default="/mnt/lynx1/datasets/audioset/eval_segments/audio_mono/audio")

    args = parser.parse_args()

    if os.path.exists(args.log_path):
        args.log_path = args.log_path.replace('.txt','_1.txt')
    
    

    #Checkpoint
    checkpoint_file = args.ckpt

    # Settings
    model_sr = 16000

    # Load score model 
    if args.mode == "storm":
        model_cls = StochasticRegenerationModel
    
    model = model_cls.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()
    
    if args.testset=='vox':
        pckl_path= './vox_test.pckl'
    elif args.testset=='lrs3':
        pckl_path= './lrs_test.pckl'
    with open(pckl_path, 'rb') as f:
        test_data = pickle.load(f) # list of dicts

    

    n_total = len(test_data)

    scores = {'pesq':[], 'stoi':[], 'estoi':[], 'si_sdr':[]}
    with open(args.log_path, 'a') as f:
        f.write(f"Evaluate separation for outputs using {args.testset},.\n")
        f.write("  pesq,     stoi,    estoi,    si_sdr\n")

    if args.testset=='vox':
        audio_dir = '/mnt/datasets/voxcelebs/voxceleb2/test/wav'
        video_dir = '/mnt/datasets/voxcelebs/voxceleb2/test/mp4'
            
    elif args.testset=='lrs3':
        audio_dir = '/mnt/datasets/lip_reading/lrs3/test'
        video_dir = '/mnt/datasets/lip_reading/lrs3/test'

    
    
    with open('/mnt/bear2/users/syun/audioset_test.txt', 'r') as f:
        audioset_list = f.readlines()
        audioset_list = [os.path.join(args.noise_path, x.strip()) for x in audioset_list]
    noise_list = audioset_list[:n_total]

    count=0

    for iden_idx, iden_dict in enumerate(tqdm.tqdm(test_data, dynamic_ncols=True)):
        noise_path = noise_list[iden_idx]

        if args.testset=='vox':
            iden1, iden2 = iden_dict.values()
            audio1_path = os.path.join(audio_dir, iden1+'.wav')
            audio2_path = os.path.join(audio_dir, iden2+'.wav')
            video1_path = os.path.join(video_dir, iden1+'.mp4')
            video2_path = os.path.join(video_dir, iden2+'.mp4')

        elif args.testset=='lrs3':
            iden1, iden2 = iden_dict.values()
            audio1_path = os.path.join(audio_dir, iden1)#+'.wav')
            audio2_path = os.path.join(audio_dir, iden2)#+'.wav')
            video1_path = os.path.join(video_dir, iden1[:-4]+'.mp4')
            video2_path = os.path.join(video_dir, iden2[:-4]+'.mp4')

        clean1, start_frame1 = load_audio_vox(audio1_path, max_len=int(16000 * 2.04), sample_rate=model_sr) #2.04
        clean2, start_frame2 = load_audio_vox(audio2_path, max_len=int(16000 * 2.04), sample_rate=model_sr)
        noise, _ = load_audio_vox(noise_path, max_len=int(16000 * 2.04), sample_rate=model_sr)

        if clean1 is None or clean2 is None or noise is None:
            continue
        

        visualFeature1 = videocap(video1_path, start_frame1)
        visualFeature2 = videocap(video2_path, start_frame2)

        clean1_n = activelev(clean1)
        clean2_n = activelev(clean2)
        noise_n = activelev(noise)
        noisy1 = clean1_n + noise_n
        noisy2 = clean2_n + noise_n
        
        t = np.random.normal() * 0.5 + 0.9
        lower=0.3
        upper=0.99
        if t < lower or t > upper:
            t = np.random.uniform(lower, upper) 
        scale = t

        max_amp = np.max(np.abs([clean1_n, clean2_n, noisy1, noisy2]))
        mix_scale = 1/max_amp*scale
        clean1 = clean1_n * mix_scale
        clean2 = clean2_n * mix_scale
        mix1 = noisy1 * mix_scale
        mix2 = noisy2 * mix_scale

        x1 = np.expand_dims(clean1, 0)#torch.Tensor(np.expand_dims(clean1, 0))
        x2 = np.expand_dims(clean2, 0)#torch.Tensor(np.expand_dims(clean2, 0))
        y1 = np.expand_dims(mix1, 0)
        y2 = np.expand_dims(mix2, 0)
        visualFeature1 = torch.Tensor(visualFeature1).cuda()
        visualFeature2 = torch.Tensor(visualFeature2).cuda()

        visualFeatures = [visualFeature1, visualFeature2]
        gt_list = [x1, x2]
        mix_list = [y1, y2]

        _pesq, _si_sdr, _estoi, _stoi = 0., 0., 0., 0.
        for idx, visfeat in enumerate(visualFeatures):
            x = gt_list[idx]
            y = torch.Tensor(mix_list[idx]).cuda()
            x_hat,_ = model.enhance(y, context = visfeat)
            
            if x_hat.ndim == 1:
                x_hat = x_hat.unsqueeze(0)
           
            if x.ndim == 1:
                x = x
                x_hat = x_hat.cpu().numpy()
                y = y.cpu().numpy()
            else: #eval only first channel
                x = x[0]
                x_hat = x_hat[0].cpu().numpy()
                y = y[0].cpu().numpy()
            

            _si_sdr += si_sdr(x, x_hat)
            _pesq += pesq(16000, x, x_hat, 'wb') 
            _estoi += stoi(x, x_hat, 16000, extended=True)
            _stoi += stoi(x, x_hat, 16000, extended=False)
            
        pesq_score = _pesq/2
        stoi_score = _stoi/2
        estoi_score = _estoi/2
        si_sdr_score = _si_sdr/2
        scores['pesq'].append(pesq_score)
        scores['stoi'].append(stoi_score)
        scores['estoi'].append(estoi_score)
        scores['si_sdr'].append(si_sdr_score)
        output_file = open(args.log_path,'a+')
        output_file.write("%3f, %3f, %3f, %3f\n" % (pesq_score, stoi_score, estoi_score, si_sdr_score))
        output_file.close()

    avg_metrics = {}
    for metric, values in scores.items():
        avg_metric = sum(values)/len(values)
        print(f"{metric}: {avg_metric}")
        avg_metrics[metric] = avg_metric

    output_file = open(args.log_path, 'a+')
    for metric, avg_metric in avg_metrics.items():
        output_file.write("%s: %3f\n" % (metric, avg_metric))
    output_file.close()
    print(f"Finished evaluating for {args.ckpt}.")


if __name__=='__main__':
    main()