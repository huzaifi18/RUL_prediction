# A Hybrid CNN-LSTM for Battery Remaining Useful Life Prediction with Charging Profiles Data
[![DOI:10.1145/3575882.3575903](http://img.shields.io/badge/DOI-10.1145/3575882.3575903-1481c1.svg)](https://doi.org/10.1145/3575882.3575903)

+ Battery RUL prediction using data-driven method based on a hybrid deep model of Convolutional Neural Networks (CNN) and Long-Short Term Memory (LSTM).
+ CNN and LSTM are used to extract features from multiple measurable data in parallel.
+ CNN extracts features of multi-channel charging profiles, whereas LSTM extracts features of historical capacity data of discharging profiles which related to time dependency.
+ This repository provides the code for training in python. It also provides trained model in saved/model/, which can be used as a baseline for your own projects.
---