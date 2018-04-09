# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:48:51 2017

@author: wayne_chen
"""
import numpy as np
import os
from python_speech_features import mfcc, sigproc, get_filterbanks, fbank
import wavio
import scipy
import re
import math
from functions import demfcc

def load_data (window,stride,feat_size,data_set,path1,path2,path3,feature_extraction,nfilt,lowfreq,preemph,sample_rate,nfft,directory):


    # type: (object, object, object, object, object, object, object, object, object, object, object, object, object, object) -> object
    """

    :rtype: object
    """

    class data:
        def __init__(self, X, Y,file_path):
            self.x = X
            self.y = Y
            self.file_path = file_path    
    D = data([],[],[])
    tmp = []
    path = data_set
    ext = '.wav'
    for tops, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(os.path.join(tops, f))[1] == ext:
                a = os.path.split(os.path.join(tops, f))[0]
                c = os.path.split(a)[1]
                a = os.path.split(a)[0]
                b = os.path.split(a)[1]
                a = os.path.split(a)[0]
                a = os.path.split(a)[1]
                if (a == path1)and(b == path2)and(c == path3):
                    tmp = feature_extract(os.path.join(tops, f),feature_extraction,window,stride,feat_size,nfilt,nfft,lowfreq,preemph,sample_rate)
                    num_samples = tmp.shape[0]
                    D.x.append(tmp)
                    D.file_path.append(os.path.join(tops, f))
                    if (path3 == 'NG'):
                        tmp = np.ones([tmp.shape[0], tmp.shape[1]], dtype=np.int32)
                        D.y.append(tmp)
                    else:
                        tmp = np.zeros([tmp.shape[0], tmp.shape[1]], dtype=np.int32)
                        D.y.append(tmp)
    if D.x:
        print(np.concatenate(D.x).shape)
        np.save(directory+'/'+path1+'_'+path2+'_'+path3+'_x', np.concatenate(D.x))
    if D.y:
        np.save(directory+'/'+path1+'_'+path2+'_'+path3+'_y', np.concatenate(D.y))
    del D,tmp


def feature_extract(path,feature_extraction,window,stride,feat_size,nfilt,nfft,lowfreq,preemph,samplerate):
    print(path)
    tmp=[]
    highfreq=samplerate/2
    appendEnergy=True
    ceplifter=nfilt
    w = wavio.read(path)
    normalized_w = w.data/(2**16./2)
    if ('Hamming'  in feature_extraction):
        winfunc = lambda x:np.hamming(x)
    else:
        winfunc = lambda x:np.ones((x,))
#    取abs_fft
    if ('mfcc' in feature_extraction):
        print('mfcc')
        tmp = demfcc.mfcc_(signal=normalized_w,samplerate=samplerate,winlen=window*0.001,winstep=stride*0.001,numcep=feat_size,nfilt=nfilt,nfft=nfft,lowfreq=lowfreq,highfreq=highfreq,preemph=preemph,ceplifter=ceplifter,appendEnergy=appendEnergy,winfunc=winfunc)
    elif('fft_sam' in feature_extraction):
        print('fft_sam')
        tmp = demfcc.fft_sam(signal=normalized_w,samplerate=samplerate,winlen=window*0.001,winstep=stride*0.001,nfft=nfft,preemph=preemph,winfunc=winfunc)
    elif ('log_abs_mel' in feature_extraction):
        print('log_abs_mel')
        tmp = demfcc.log_abs_mel(signal=normalized_w, samplerate=samplerate, winlen=window * 0.001,
                                 winstep=stride * 0.001,
                                 nfilt=nfilt, nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph,
                                 winfunc=winfunc)
    if ('recurr' in feature_extraction):
        pair_len = int(re.search('@(.*?)_', feature_extraction).group(1))
        pair_step = int(re.search('~(.*?)_', feature_extraction).group(1))
        numpairs = math.floor((tmp.shape[0]-pair_len)/pair_step)+1
        indices = np.tile(np.arange(0, pair_len), (numpairs, 1)) + np.tile(
            np.arange(0, numpairs * pair_step, pair_step), (pair_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        tmp = tmp[indices]
    return np.float32(tmp)
