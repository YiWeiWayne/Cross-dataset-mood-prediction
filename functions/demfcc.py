# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 13:56:33 2017

@author: wayne_chen
"""

import numpy
from python_speech_features import fbank, sigproc,lifter
from scipy.fftpack import dct


def fft_sam(signal,samplerate=16000,winlen=0.08,winstep=0.04,nfft=2048,preemph=0.97,
         winfunc=lambda x:numpy.ones((x,))):
        signal = sigproc.preemphasis(signal, preemph)
        frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
        feat = numpy.float32(numpy.absolute(numpy.fft.fft(frames, nfft)))[:, 0:int(numpy.floor(nfft/2))+1]
        for i in range(0,len(feat)):
                feat[i,1:] = feat[i,1:]-feat[i,:-1]
        return feat


def abs_fft(signal,samplerate=16000,winlen=0.08,winstep=0.04,nfft=2048,preemph=0.97,
         winfunc=lambda x:numpy.ones((x,))):
        signal = sigproc.preemphasis(signal, preemph)
        frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
        feat = sigproc.magspec(frames,nfft)
        return feat


def abs_preemph_fft(signal, samplerate=16000, winlen=0.08, winstep=0.04, nfft=2048, preemph=0.97,
            winfunc=lambda x: numpy.ones((x,))):
        signal = sigproc.preemphasis(signal, preemph)
        frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
        feat = sigproc.magspec(frames, nfft)
        feat = sigproc.preemphasis(feat, 1)
        return feat


def power_(signal,samplerate=16000,winlen=0.08,winstep=0.04,nfft=2048,preemph=0.97,
         winfunc=lambda x:numpy.ones((x,))):        
        signal = sigproc.preemphasis(signal,preemph)
        frames = sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
        feat = sigproc.powspec(frames,nfft)
        return feat


def tri_log(signal,samplerate=16000,winlen=0.08,winstep=0.04,
         nfilt=39,nfft=2048,lowfreq=12.5,highfreq=None,preemph=0.97,winfunc=lambda x:numpy.ones((x,))):
        feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
        feat = numpy.log(feat)
        return feat


def log_abs_mel(signal, samplerate=16000, winlen=0.08, winstep=0.04,
            nfilt=39, nfft=2048, lowfreq=12.5, highfreq=None, preemph=0.97, winfunc=lambda x: numpy.ones((x,))):
        feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
        feat = numpy.float32(numpy.log(numpy.absolute(feat)))
        return feat


def dct_(signal,samplerate=16000,winlen=0.08,winstep=0.04,numcep=39,
         nfilt=39,nfft=2048,lowfreq=12.5,highfreq=None,preemph=0.97, winfunc=lambda x:numpy.ones((x,))):
        feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
        feat = numpy.log(feat)
        feat = dct(feat, n=max(numcep,feat.shape[1]), type=2, axis=1, norm='ortho')[:,:numcep]
        return feat


def lift(signal,samplerate=16000,winlen=0.08,winstep=0.04,numcep=39,
         nfilt=39,nfft=2048,lowfreq=12.5,highfreq=None,preemph=0.97,ceplifter=39,winfunc=lambda x:numpy.ones((x,))):
        feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
        feat = numpy.log(feat)
        feat = dct(feat, n=max(numcep,feat.shape[1]), type=2, axis=1, norm='ortho')[:,:numcep]
        feat = lifter(feat,ceplifter)
        return feat


def mfcc_(signal,samplerate=16000,winlen=0.08,winstep=0.04,numcep=39,
         nfilt=39,nfft=2048,lowfreq=12.5,highfreq=None,preemph=0.97,ceplifter=39,appendEnergy=True,winfunc=lambda x:numpy.ones((x,))):
        feat,energy = fbank(signal,samplerate,winlen,winstep,nfilt,nfft,lowfreq,highfreq,preemph,winfunc)
        feat = numpy.log(feat)
        feat = dct(feat, n=max(numcep,feat.shape[1]), type=2, axis=1, norm='ortho')[:,:numcep]
        feat = lifter(feat,ceplifter)
        if appendEnergy: feat[:,0] = numpy.log(energy)# replace first cepstral coefficient with log of frame energy
        return feat