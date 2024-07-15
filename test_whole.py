import numpy as np
import glob
import tqdm
import torch
import os
from argparse import ArgumentParser
import time
from pypapi import events, papi_high as high
import soundfile as sf
from sgmse.model import StochasticRegenerationModel
import matplotlib.pyplot as plt
from sgmse.util.other import *

from sgmse.util.other import si_sdr, pad_spec
from pesq import pesq
from pystoi import stoi

import cv2
import pickle
import librosa

EPS_LOG = 1e-10
sr = 16000
hop_length = 2.04


def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

    
def prep_audio(audio1_path, audio2_path,noise_path, sample_rate=sr):
    clean1, sample_rate = librosa.load(audio1_path, sr=sr)
    clean2, sample_rate = librosa.load(audio2_path, sr=sr)
    noise, _ = librosa.load(noise_path, sr=sr)


    min_len_1 = min(len(clean1), len(noise))
    min_len_2 = min(len(clean2), len(noise))

    clean1_n = activelev(clean1[:min_len_1])
    clean2_n = activelev(clean2[:min_len_2])
    noise_n = activelev(noise[:min_len_1])
    noise_n2 = activelev(noise[:min_len_2])


    noisy1 = clean1_n + noise_n
    noisy2 = clean2_n + noise_n2
    
    t = np.random.normal() * 0.5 + 0.9
    lower=0.3
    upper=0.99
    if t < lower or t > upper:
        t = np.random.uniform(lower, upper) 
    scale = t

    max_amp = np.max(np.abs([clean1_n, noisy1]))
    max_amp_2 = np.max(np.abs([ clean2_n,  noisy2]))
    mix_scale = 1/max_amp*scale
    mix_scale_2 = 1/max_amp_2*scale
    clean1 = clean1_n * mix_scale
    clean2 = clean2_n * mix_scale_2
    mix1 = noisy1 * mix_scale
    mix2 = noisy2 * mix_scale_2

    return [clean1, clean2], [mix1, mix2] # long audio


def videocap(path, start_frame): # for VoxCeleb2
    vid_start = int(start_frame/16000*25)
    cap = cv2.VideoCapture(path)
    vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fm_num= vidlength

    if cap.isOpened():
        frames=[]
        for i in range(vid_start+fm_num):
            ret, img = cap.read()
            if i<vid_start:
                continue
            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (112,112))
                frames.append(img)
            else:
                if len(frames)==0:
                    print(path, " is not opened…")
                    import pdb; pdb.set_trace()
                    return None
                frames = np.array(frames)
                #print(frames.shape)
                
                try:
                    frames = np.pad(frames, ((0, vid_start+fm_num-i), (0,0), (0,0)), 'wrap')
                except:
                    import pdb; pdb.set_trace()
                assert frames.shape == (fm_num, 112, 112), "padding is set wrong"
                return frames
        frames = np.array(frames)
        return frames # (51, H, W)
    else:
        print(path, " is not opened…")
        return None

def prep_video(video1_path, video2_path, start_frame):
    visualFeature1 = videocap(video1_path, start_frame)
    visualFeature2 = videocap(video2_path, start_frame)
    if visualFeature1 is None:
        print(f"{video1_path} is invalid!!! ")
        return None
    elif visualFeature2 is None:
        print(f"{video2_path} is invalid!!! ")
        return None
    visualFeature1 = torch.Tensor(visualFeature1).cuda()
    visualFeature2 = torch.Tensor(visualFeature2).cuda()
    return [visualFeature1, visualFeature2]


def activelev(data):
    max_amp = np.std(data)
    return data/max_amp

def main():
    # Tags
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument("--ckpt", type=str, default='/mnt/bear2/users/cyong/log_FM/mode=regen-joint-training_sde=OUVESDE_score=ncsnpp_crossatt_data=voxceleb2_SE_ch=1/version_66/checkpoints/last.ckpt')
        parser_.add_argument("--mode", type=str, default="storm")
        parser_.add_argument('--log_path', type=str, default='./test_whole_ori_vox.txt')
        parser_.add_argument("--testset", default='vox', type=str, choices=['lrs3', 'vox'])
        parser_.add_argument("--noise_path", default="/mnt/lynx1/datasets/audioset/eval_segments/audio_mono/audio")
        parser_.add_argument("--data_dir", default='/mnt/datasets/voxcelebs/voxceleb2/', type=str, help='path of data directory corresponding to the testset choice')  # LRS3: /mnt/datasets/lip_reading/lrs3/ 

    args = parser.parse_args()

    checkpoint_file = args.ckpt
    model_sr = 16000
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
        test_data = pickle.load(f)

    n_total = len(test_data)

    scores = {'pesq':[], 'stoi':[], 'estoi':[], 'si_sdr':[], 'pesq_den':[], 'stoi_den':[], 'estoi_den':[], 'si_sdr_den':[]}
    with open(args.log_path, 'a') as f:
        f.write(f"Evaluate separation for outputs using {args.testset}\n")
        f.write("  pesq,     stoi,    estoi,    si_sdr,    pesq_den,  stoi_den,  estoi_den,  si_sdr_den\n")
    
    if args.testset=='vox':
        audio_dir = os.path.join(args.data_dir,'test/wav')
        video_dir = os.path.join(args.data_dir, 'test/mp4')
            
    elif args.testset=='lrs3':
        audio_dir = os.path.join(args.data_dir,'test')
        video_dir = os.path.join(args.data_dir,'test')


    with open('/mnt/bear2/users/syun/audioset_test.txt', 'r') as f:
        audioset_list = f.readlines()
        audioset_list = [os.path.join(args.noise_path, x.strip()) for x in audioset_list]
    noise_list = audioset_list[:n_total]


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
                audio1_path = os.path.join(audio_dir, iden1)
                audio2_path = os.path.join(audio_dir, iden2)
                video1_path = os.path.join(video_dir, iden1[:-4]+'.mp4')
                video2_path = os.path.join(video_dir, iden2[:-4]+'.mp4')

            gt_list_long, mix_long = prep_audio(audio1_path, audio2_path, noise_path)
            audio_length = len(mix_long[0])
            audio_length_2 = len(mix_long[1])

            sliding_window_start = 0
            sep_audio1 = np.zeros((audio_length))
            sep_audio2 = np.zeros((audio_length_2))
            sep_audio_list = [sep_audio1, sep_audio2]
            den_audio1 = np.zeros((audio_length))
            den_audio2 = np.zeros((audio_length_2))
            den_audio_list = [den_audio1, den_audio2]


            gt_list =  gt_list_long
            mix = mix_long
            visualFeatures = prep_video(video1_path, video2_path, sliding_window_start)
            if visualFeatures is None:
                continue
            
            spec_list =[]
            for i, visfeat in enumerate(visualFeatures):
                x = gt_list[i].squeeze()
                y = torch.Tensor(np.expand_dims(mix[i], 0)).cuda()

                x_spec = model._stft(torch.from_numpy(x))
                x_hat_spec, y_spec, y_den_spec, T_orig, norm_factor = model.enhance(y, context = visfeat, return_stft=True)
                    
                x_hat = model.to_audio(x_hat_spec, T_orig)
                x_hat = x_hat * norm_factor
                x_hat = x_hat.squeeze()
                y_den = model.to_audio(y_den_spec, T_orig)
                y_den = y_den * norm_factor
                y_den = y_den.squeeze()
                    
                if x.ndim == 1:
                    x_hat = x_hat.cpu().numpy()
                    y_den = y_den.cpu().numpy()
                else:
                    x_hat = x_hat[0].cpu().numpy()
                    y_den = y_den[0].cpu().numpy()
                sep_audio_list[i][sliding_window_start:] += x_hat
                den_audio_list[i][sliding_window_start:] += y_den
                spec_list.append(x_spec)
                spec_list.append(y_spec)
                spec_list.append(x_hat_spec)

            
            
            sep_audio1, sep_audio2 = sep_audio_list
            pred_list = [sep_audio1, sep_audio2]
            den_audio_list=[den_audio1,den_audio2]


            
            
            

            # calculate metric for full audio
            _pesq, _si_sdr, _estoi, _stoi, _pesq_den, _si_sdr_den, _estoi_den, _stoi_den = 0., 0., 0., 0., 0., 0., 0., 0.
            for x, x_hat, y_den in zip(gt_list_long, pred_list, den_audio_list):
                try:
                    _pesq += pesq(16000, x, x_hat, 'wb') 
                except:
                    print('skip count')
                    break
                _si_sdr += si_sdr(x, x_hat)
                _estoi += stoi(x, x_hat, 16000, extended=True)
                _stoi += stoi(x, x_hat, 16000, extended=False)
                _si_sdr_den += si_sdr(x, y_den)
                _pesq_den += pesq(16000, x, y_den, 'wb') 
                _estoi_den += stoi(x, y_den, 16000, extended=True)
                _stoi_den += stoi(x, y_den, 16000, extended=False)
            if _stoi==0:
                continue
            else:
                pesq_score = _pesq/2
                stoi_score = _stoi/2
                estoi_score = _estoi/2
                si_sdr_score = _si_sdr/2
                pesq_score_den = _pesq_den/2
                stoi_score_den = _stoi_den/2
                estoi_score_den = _estoi_den/2
                si_sdr_score_den = _si_sdr_den/2
                scores['pesq'].append(pesq_score)
                scores['stoi'].append(stoi_score)
                scores['estoi'].append(estoi_score)
                scores['si_sdr'].append(si_sdr_score)
                scores['pesq_den'].append(pesq_score_den)
                scores['stoi_den'].append(stoi_score_den)
                scores['estoi_den'].append(estoi_score_den)
                scores['si_sdr_den'].append(si_sdr_score_den)
                output_file = open(args.log_path,'a+')
                output_file.write("%3f, %3f, %3f, %3f, %3f, %3f, %3f, %3f\n" % (pesq_score, stoi_score, estoi_score, si_sdr_score, pesq_score_den, stoi_score_den, estoi_score_den, si_sdr_score_den))
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