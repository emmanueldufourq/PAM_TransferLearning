# Transfer learning for passive acoustic monitoring

In this study we describe the implementation of transfer learning to create bioacoustic classification models as a means to reduce the complexity of implementing deep learning algorithms from scratch. We compared 12 modern CNN architectures across 4 datasets and focus on scarce data issues when attempting to create a passive acoustic binary classification model with as few as 25 verified examples. The datasets contained vocalisations of: the critically endangered Hainan gibbon (Nomascus hainanus), the critically endangered black-and-white ruffed lemur (Varecia variegata), the vulnerable Thyolo alethe (Chamaetylas choloensis), and the Pin-tailed whydah (Vidua macroura). 

Our goal was to demonstrate that transfer learning is a suitable and accessible approach to implementing CNNs which require less human expert knowledge in machine learning.

This code accompanies the paper ``Passive Acoustic Monitoring and Transfer Learning''.

# Authors
Emmanuel Dufourq, Carly Batist, Ruben Foquet and Ian Durbach

<hr>

# Requirements
Developed and tested with Python 3.8

Install all requirements using pip install -r requirements.txt

tensorflow==2.7.0
SoundFile==0.10.3.post1
scikit-learn==1.0.2
scipy==1.8.0
numpy==1.18.5
jupyter==1.0.0
pandas==1.4.1
librosa==0.8.0
yattag==1.14.0
matplotlib==3.3.3
Keras==2.7.0

<hr>