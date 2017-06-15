#!/usr/bin/env python

"""
    speechworker.py
"""

import audioBasicIO as aIO
import audioFeatureExtraction as aF
import audioTrainTest as aT

class SpeechWorker(object):
    
    def __init__(self, model_name='data/svmSM', model_type='svm'):
        assert(model_name == 'data/svmSM')
        assert(model_type == 'svm')
        
        [self.classifier, self.model_mean, self.model_sd, self.class_names, 
            self.mt_win, self.mt_step, self.st_win, self.st_step, _] = aT.loadSVModel(model_name)
    
    def predict(self, input_file):
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
        return dict(zip(self.class_names, p))

if __name__ == "__main__":
    sw = SpeechWorker()
    sw.predict("./data/speechEmotion/59.wav")
