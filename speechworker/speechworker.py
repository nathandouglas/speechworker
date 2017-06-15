#!/usr/bin/env python

"""
    speechworker.py
    
    data/svmSM is a pretrained model from `pyAudioAnalysis`
"""

import os
import pyAudioAnalysis.audioBasicIO as aIO
import pyAudioAnalysis.audioFeatureExtraction as aF
import pyAudioAnalysis.audioTrainTest as aT

class SpeechWorker(object):
    model_type = 'svm'
    def __init__(self, model_path=None):
        assert(self.model_type == 'svm')
        
        if not model_path:
            ppath = os.path.join(os.environ['HOME'], '.speechworker')
            model_path = os.path.join(ppath, 'models/svmSM')
        
        [self.classifier, self.model_mean, self.model_sd, self.class_names, 
            self.mt_win, self.mt_step, self.st_win, self.st_step, _] = aT.loadSVModel(model_path)
    
    def __call__(self, input_file):
        (frame_rate, x) = aIO.readAudioFile(input_file)
        [feats, s] = aF.mtFeatureExtraction(
            aIO.stereo2mono(x), 
            frame_rate, 
            self.mt_win * frame_rate, 
            self.mt_step * frame_rate, 
            round(frame_rate * self.st_win), 
            round(frame_rate * self.st_step)
        )
        feats = feats.mean(axis=1)
        feats = (feats - self.model_mean) / self.model_sd
        
        p = self.classifier.predict_proba(feats.reshape(1, -1))[0]
        out = dict(zip(self.class_names, p))
        out.update({
            "_frame_rate" : frame_rate,
            "_duraction_seconds" : float(x.shape[0]) / frame_rate
        })
        return out