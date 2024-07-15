import cv2

from os.path import join
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import torch.nn.functional as F
import h5py
import json
import librosa
import csv
from scipy.io import wavfile
import random

SEED = 20
np.random.seed(SEED)

def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")

def activelev(data):
    max_amp = np.std(data) 
    if max_amp==0:
        return 0
    return data/max_amp



def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples

def load_audio_vox_vv(file_path, max_len, sample_rate, tensor=False): 
    _, audio = wavfile.read(file_path)
    
    audio = audio/32768
    audiosize = audio.shape[0] 
    if audiosize < max_len:
        shortage = max_len - audiosize
        audio = np.pad(audio, (0, shortage), 'wrap')
    else:
        audio = audio[0:max_len]

    if audio.shape[0]!=max_len:
        raise Exception('Length error occur while loading the audio file')

    return torch.FloatTensor(audio) if tensor else audio



class Specs(Dataset):
    def __init__(
        self, data_dir, subset, dummy, shuffle_spec, num_frames, format, use_sync_encoder,
        audiovisual=True,
        normalize_audio=True, spec_transform=None, stft_kwargs=None, spatial_channels=1, 
        return_time=False,
        **ignored_kwargs
    ):
        self.data_dir = data_dir
        self.subset = subset
        self.use_sync_encoder = use_sync_encoder

        if audiovisual and format != 'voxceleb2':
            faceFolderPath = os.path.join(data_dir, 'face_input', subset)
            self.faceFiles = glob("%s/*.jpg"%faceFolderPath)
        self.audiovisual = audiovisual
        self.format = format
        self.spatial_channels = spatial_channels
        self.return_time = return_time
        
        
        if format == 'voxceleb2_SE':
            if subset == 'train':
                self.data_path = "/mnt/datasets/voxcelebs/voxceleb2/dev/wav/" #data_path
                self.sample_num = len(glob(os.path.join(self.data_path, '*/*/*.wav')))
                self.noise_path = "/mnt/lynx1/datasets/audioset/unbalanced_train_segments/audio_mono"
                with open('/mnt/bear2/users/syun/audioset_train.txt') as f:
                    noise_list = f.readlines()
                    noise_list = [os.path.join(self.noise_path, filename.strip()) for filename in noise_list]
                self.noise_list = np.random.choice(noise_list,self.sample_num)
                self.dynamic_mixing = False
            else:
                self.data_path = "/mnt/datasets/voxcelebs/voxceleb2/test/wav/"
                self.noise_path = "/mnt/lynx1/datasets/audioset/eval_segments/audio_mono/audio"
                self.sample_num = 1000 
                with open('/mnt/bear2/users/syun/audioset_test.txt') as f:
                    noise_list = f.readlines()
                    noise_list = [os.path.join(self.noise_path, filename.strip()) for filename in noise_list]
                self.noise_list = noise_list[:self.sample_num]
                self.dynamic_mixing = False
            self.sample_rate = 16000 #sample_rate
            self.chunk_size = int(16000 * 2.04) #chunk_size

            self.spk_table = self.spk_table_generator(self.data_path)
            self.vid_table = self.vid_table_generator(self.data_path)
            self.sample_list = self.sample_generator_fromvid(self.sample_num)

        
        elif format == 'voxceleb2_SS': # or "voxceleb2_ssrd":
            if subset == 'train':
                self.data_path = "/mnt/datasets/voxcelebs/voxceleb2/dev/wav/" #data_path
                self.noise_path = "/mnt/datasets/voxcelebs/voxceleb2/dev/wav/"
                self.sample_num = len(glob(os.path.join(self.data_path, '*/*/*.wav')))
                self.dynamic_mixing = False
                self.noise_list = glob(os.path.join(self.noise_path,'*/*/*.wav')) if self.dynamic_mixing else np.random.choice(glob(os.path.join(self.noise_path,'*/*/*.wav')),self.sample_num)
            else:
                self.data_path = "/mnt/datasets/voxcelebs/voxceleb2/test/wav/"
                self.noise_path = "/mnt/datasets/voxcelebs/voxceleb2/test/wav/"
                self.sample_num = 5000 #2000
                self.dynamic_mixing = False
                # noise_list: changed from '*.wav'(for WHAM) to '*/*.wav' for DEMAND
                self.noise_list = sorted(glob(os.path.join(self.noise_path,'*/*/*.wav')))[:self.sample_num] if self.dynamic_mixing else np.random.choice(glob(os.path.join(self.noise_path,'*/*/*.wav')),self.sample_num)

            self.snr_list = [0,5,10] #snr_list
            self.sample_rate = 16000 #sample_rate
            self.chunk_size = int(16000 * 2.04) #chunk_size

            self.spk_table = self.spk_table_generator(self.data_path)
            self.vid_table = self.vid_table_generator(self.data_path)
            self.sample_list = self.sample_generator_fromvid(self.sample_num)
        

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize_audio = False #normalize_audio
        self.spec_transform = spec_transform

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def _open_hdf5(self):
        self.meta_data = json.load(open(sorted(glob(join(self.data_dir, f"*.json")))[-1], "r"))
        self.prep_file = h5py.File(sorted(glob(join(self.data_dir, f"*.hdf5")))[-1], 'r')

    def spk_table_generator(self,data_path):
        id_list = os.listdir(self.data_path)
        spk_table = {}
        for id in id_list:
            wav_list = glob(os.path.join(self.data_path,id,"*","*.wav"))
            for wav_path in wav_list:
                video_nm = os.path.basename(os.path.dirname(wav_path))
                if id not in spk_table:
                    spk_table[id] = {}
                if not video_nm in spk_table[id]:
                    spk_table[id][video_nm] = []
                spk_table[id][video_nm].append(wav_path)

        return spk_table

    def vid_table_generator(self,data_path):
        id_list = os.listdir(self.data_path)
        vid_table = {}
        vid_list = {}
        for id in id_list:
            #import pdb; pdb.set_trace()
            vid_list = os.listdir(os.path.join(self.data_path,id))
            for vid in vid_list:
                vid_table[vid]= glob(os.path.join(self.data_path,id,vid,"*.wav"))
                if len(vid_table[vid])==0:
                    raise ValueError(os.path.join(data_path, id, vid), os.listdir(os.path.join(data_path,id,vid)))

        return vid_table


    def sample_generator_fromvid(self, sample_num):
        sample_list = []

        replace = (sample_num >= len(self.vid_table))
        selected_vids = np.random.choice(list(self.vid_table.keys()), sample_num, replace=replace)
        
        for vid in selected_vids:
            sample_list.append(np.random.choice(self.vid_table[vid], 1)[0])
        return sample_list



    def sample_data(self, id):
        selected_vid = np.random.choice(list(self.spk_table[id].keys()),1)[0]
        selected_sample = np.random.choice(self.spk_table[id][selected_vid],1)[0]

        return selected_sample


    def id_extractor(self, path):
        return os.path.basename(os.path.dirname(os.path.dirname(path)))


    def videocap(self,path,start_frame, for_sync=False): # for VoxCeleb2
        # start_frame is set according to the audio frames
        vid_start = int(start_frame//16000*25)
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            frames=[]
            for i in range(vid_start+51):
                ret, img = cap.read()
                if i<vid_start:
                    continue

                if ret:
                    if self.use_sync_encoder:# or for_sync:
                        img = cv2.resize(img, (224,224))
                    elif for_sync:
                        img = cv2.resize(img, (96,96))
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, (112,112))
                    frames.append(img)
                else:
                    if i-vid_start<30:
                        return None
                    ##(path, " doesn't have enough frames. Remaining", 51-i, "frames are padded with 'wrap' mode")
                    frames = np.array(frames)
                    
                    if self.use_sync_encoder:
                        frames = np.pad(frames, ((0, vid_start+51-i), (0,0), (0,0), (0,0)), 'wrap')
                        assert frames.shape == (51, 224, 224, 3), "padding is set wrong"
                    elif for_sync:
                        frames = np.pad(frames, ((0, vid_start+51-i), (0,0), (0,0), (0,0)), 'wrap')
                        assert frames.shape == (51, 96, 96, 3)
                    else:
                        frames = np.pad(frames, ((0, vid_start+51-i), (0,0), (0,0)), 'wrap')
                        assert frames.shape == (51, 112, 112), "padding is set wrong"
                    return frames
            frames = np.array(frames)
            return frames # (51, H, W, C if exists)
        else:
            ##(path, " is not opened. Getting other video…")
            return None

    def load_audio_vox(self, file_path):
        try:
            audio, sample_rate = librosa.load(file_path, sr=self.sample_rate) # mono as default
        except:
            return None, 0
        audiosize = audio.shape[0]

        if audiosize < self.chunk_size:
            shortage = self.chunk_size - audiosize
            min_len = 16000*30//25
            if audiosize < min_len:
                ##(file_path + " is too short. Trying another video…")
                return None, 0
            audio = np.pad(audio, (0, shortage), 'wrap')
            start_frame = 0
            if np.all((audio==0)):
                ##(file_path+" loaded as zero array. Trying another video…")
                return None, 0
            
        else:
            if self.subset == 'train':
                start_frame = random.randrange(0, audiosize - self.chunk_size)
            else:
                #start_frame = (audiosize - self.chunk_size) // 2 # if 0, 'uhm' is often included. So, let's choose the medium
                start_frame = 0
     
            #start_frame=0
            audio = audio[start_frame:start_frame+self.chunk_size]
            if np.all((audio==0)):
                return None, 0
        
        return audio, start_frame


    def __getitem__(self, i, raw=False):
        
        if self.format == 'voxceleb2_SS' or self.format=="voxceleb2_ssrd" or self.format =="voxceleb2_SE":
            videopath = self.sample_list[i].replace('/wav/','/mp4/').replace('.wav','.mp4')

            gt, start_frame = self.load_audio_vox(self.sample_list[i])
            visualFeatures = self.videocap(videopath, start_frame)
            while visualFeatures is None or gt is None:
                i = random.randrange(len(self.sample_list))
                gt, start_frame = self.load_audio_vox(self.sample_list[i])
                videopath = self.sample_list[i].replace('/wav/','/mp4/').replace('.wav','.mp4')
                visualFeatures = self.videocap(videopath, start_frame)
                
            noise_idx = i
            selected_noise = np.random.choice(self.noise_list,1)[0] if self.dynamic_mixing else self.noise_list[i]
            noise, _ = self.load_audio_vox(selected_noise)
            while noise is None or (selected_noise == self.sample_list[i]):
                noise_idx += 1
                if noise_idx > self.sample_num:
                    noise_idx = 0
                selected_noise = np.random.choice(self.noise_list,1)[0] if self.dynamic_mixing else self.noise_list[noise_idx]
                noise, _ = self.load_audio_vox(selected_noise)
            
            
            clean_n = activelev(gt)
            noise_n = activelev(noise)
            clean = clean_n #* clean_weight make same weight 0.5
            noise = noise_n #* noise_weight
            noisy = clean + noise
            
            t = np.random.normal() * 0.5 + 0.9 
            lower=0.3
            upper=0.99
            if t < lower or t > upper:
                t = np.random.uniform(lower, upper) 
            scale = t 


            try:
                abs = np.abs([noise, clean, noisy]).any()
            except:
                print('error occurs')
                abs = np.abs(noisy)
            max_amp = np.max(abs)
            mix_scale = 1/max_amp*scale
            gt = clean * mix_scale
            mix = noisy * mix_scale
            

            x = torch.Tensor(np.expand_dims(gt, 0))  # torch.Size([1, 32640])
            if self.subset != 'train':
                y = torch.Tensor(np.expand_dims(mix, 0)) # torch.Size([1, 32640])
            else:
                if self.format == "voxceleb2_SS" or self.format == "voxceleb2_SE":
                    y = torch.Tensor(np.expand_dims(mix, 0)) # torch.Size([1, 32640])
                else: 
                    if torch.rand(1)[0] <0.3:
                        y = torch.Tensor(np.expand_dims(gt, 0)) 
                    else:
                        y = torch.Tensor(np.expand_dims(mix, 0))

        

        else:

            x, _ = load(self.clean_files[i])		 #[1,47926]	
            y, _ = load(self.noisy_files[i])		 #[1.47926]
        

        min_len = min(x.size(-1), y.size(-1))
        x, y = x[..., : min_len], y[..., : min_len] 

        if x.ndimension() == 2 and self.spatial_channels == 1:
            x, y = x[0].unsqueeze(0), y[0].unsqueeze(0) #Select first channel
        # Select channels
        assert self.spatial_channels <= x.size(0), f"You asked too many channels ({self.spatial_channels}) for the given dataset ({x.size(0)})"
        x, y = x[: self.spatial_channels], y[: self.spatial_channels]


        
        if raw:

            return x, y, visualFeatures

        normfac = y.abs().max()

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length #255*128=32640
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)


        if self.normalize_audio:
            x = x / normfac
            y = y / normfac

        if self.return_time:
            return x, y


        X = torch.stft(x, **self.stft_kwargs) #[1,32640] -> [1,256,256]
        Y = torch.stft(y, **self.stft_kwargs)

        X, Y = self.spec_transform(X), self.spec_transform(Y)


        if self.audiovisual:
            return X, Y, torch.Tensor(visualFeatures)
        else:
            return X, Y

    def __len__(self):
        if self.format=='voxceleb2_SS' or self.format =='voxceleb2_SE' or self.format=='voxceleb2_SES' or self.format=="avsLRS_SE":
            if self.dummy:
                # for debugging shrink the data set sizer
                return int(self.sample_num/10)
            else:
                return self.sample_num #self.sample_num #self.sample_num
        else:
            if self.dummy:
                # for debugging shrink the data set sizer
                return int(len(self.clean_files)/10)
            else:
                if self.format == "vctk":
                    return len(self.clean_files)//2
                else: 
                    return self.sample_num





class SpecsDataModule(pl.LightningDataModule):
    def __init__(
        self, use_sync_encoder=False, base_dir="", format="voxceleb2_SS", spatial_channels=1, batch_size=8,
        n_fft=510, hop_length=128, num_frames=256, window="hann",
        num_workers=8, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, return_time=False, **kwargs
    ):
        super().__init__()
        self.base_dir = base_dir
        self.format = format
        self.spatial_channels = spatial_channels
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.return_time = return_time
        self.use_sync_encoder = use_sync_encoder
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd,
            **self.stft_kwargs, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(self.base_dir, 'train', self.dummy, True, 
                format=self.format, spatial_channels=self.spatial_channels, 
                return_time=self.return_time, use_sync_encoder=self.use_sync_encoder, **specs_kwargs)
            self.valid_set = Specs(self.base_dir, 'valid', self.dummy, False, 
                format=self.format, spatial_channels=self.spatial_channels, 
                return_time=self.return_time, use_sync_encoder=self.use_sync_encoder, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(self.base_dir, 'test', self.dummy, False, 
                format=self.format, spatial_channels=self.spatial_channels, 
                return_time=self.return_time, use_sync_encoder=self.use_sync_encoder, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs()**e * torch.exp(1j * spec.angle())
        return spec * self.spec_factor

    def spec_back(self, spec):
        spec = spec / self.spec_factor
        if self.spec_abs_exponent != 1:
            e = self.spec_abs_exponent
            spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--format", type=str, default="voxceleb2_SE", choices=["voxceleb2_SE",  'voxceleb2_SS'], help="File paths follow the DNS data description.")
        parser.add_argument("--base_dir", type=str, default="/mnt/scratch/datasets/new_avspeech", #"/mnt/scratch/datasets/new_avspeech" , "/mnt/work2/users/cyong/storm/new_avspeech"
            help="The base directory of the dataset. Should contain `train`, `valid` and `test` subdirectories, "
                "each of which contain `clean` and `noisy` subdirectories.")
        parser.add_argument("--use_sync_encoder", action="store_true", help="enable this option when the denoiser is 'ncsnpp_crossatt_sync'") # 이거 잘 안 먹히는듯? 어떻게 고쳐야할깜
        parser.add_argument("--batch_size", type=int, default=4, help="The batch size. 32 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 256 freq bins
        parser.add_argument("--hop_length", type=int, default=128, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'sqrthann' by default.")
        parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.33, help="Factor to multiply complex STFT coefficients by.") ##### In Simon's current impl, this is 0.15 !
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5,
            help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). "
                "1 by default; set to values < 1 to bring out quieter features.")
        parser.add_argument("--return_time", action="store_true", help="Return the waveform instead of the STFT")
        
        
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True # pin_memory=self.gpu
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False # pin_memory=self.gpu
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False # pin_memory=self.gpu
        )













'''
class SpecsAndTranscriptions(Specs):

    def __init__(
        self, data_dir, subset, dummy, shuffle_spec, num_frames, format,
        **kwargs
    ):
        super().__init__(data_dir, subset, dummy, shuffle_spec, num_frames, format, **kwargs)
        if format == "timit":
            dic_correspondence_subsets = {"train": "tr", "valid": "cv", "test": "tt"}
            self.clean_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/clean/*.wav'))
            self.noisy_files = sorted(glob(join(data_dir, "audio", dic_correspondence_subsets[subset]) + '/noisy/*.wav'))
            self.transcriptions = sorted(glob(join(data_dir, "transcriptions", dic_correspondence_subsets[subset]) + '/*.txt'))
        else:
            raise NotImplementedError

    def __getitem__(self, i, raw=False):
        X, Y = super().__getitem__(i, raw=raw)
        transcription = open(self.transcriptions[i], "r").read()
        if self.format == "timit": #remove the number at the beginning
            transcription = " ".join(transcription.split(" ")[2: ])

        return X, Y, transcription

    def __len__(self):
        if self.dummy:
            return int(len(self.clean_files)/10)
        else:
            return len(self.clean_files)

class SpecsAndTranscriptionsDataModule(SpecsDataModule):

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd,
            **self.stft_kwargs, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            raise NotImplementedError
        if stage == 'test' or stage is None:
            self.test_set = SpecsAndTranscriptions(self.base_dir, 'test', self.dummy, False, 
            format=self.format, **specs_kwargs)


    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--format", type=str, default="avspeech", choices=["wsj0", "vctk", "dns", "reverb_wsj0",'avspeech'], help="File paths follow the DNS data description.")
        parser.add_argument("--base-dir", type=str, default="/data/lemercier/databases/reverb_wsj0+chime/audio")
        parser.add_argument("--batch-size", type=int, default=8, help="The batch size.")
        parser.add_argument("--num-workers", type=int, default=8, help="Number of workers to use for DataLoaders.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        return parser

'''