#!/bin/bash

mkdir -p ~/.speechworker/models/
wget https://raw.githubusercontent.com/tyiannak/pyAudioAnalysis/master/data/svmSM -O ~/.speechworker/models/svmSM
wget https://github.com/tyiannak/pyAudioAnalysis/raw/master/data/svmSM.arff -O ~/.speechworker/models/svmSM.arff
wget https://github.com/tyiannak/pyAudioAnalysis/raw/master/data/svmSMMEANS -O ~/.speechworker/models/svmSMMEANS