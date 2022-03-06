# Transfer learning for passive acoustic monitoring

In this study we describe the implementation of transfer learning to create bioacoustic classification models as a means to reduce the complexity of implementing deep learning algorithms from scratch. We compared 12 modern CNN architectures across 4 datasets and focus on scarce data issues when attempting to create a passive acoustic binary classification model with as few as 25 verified examples. The datasets contained vocalisations of: the critically endangered Hainan gibbon (Nomascus hainanus), the critically endangered black-and-white ruffed lemur (Varecia variegata), the vulnerable Thyolo alethe (Chamaetylas choloensis), and the Pin-tailed whydah (Vidua macroura). 

Our goal was to demonstrate that transfer learning is a suitable and accessible approach to implementing CNNs which require less human expert knowledge in machine learning.

This code accompanies the paper ``Passive Acoustic Monitoring and Transfer Learning''.

# Authors
Emmanuel Dufourq, Carly Batist, Ruben Foquet and Ian Durbach

<hr>

# Requirements

Install all requirements using pip install -r requirements.txt

The following packages were used in development and have been tested on Ubuntu 20.04.3 LTS running Python 3.8 (NVIDIA-SMI 470.103.01, CUDA Version: 11.2, cudnn: 8.1.1)

* tensorflow==2.7.0
* SoundFile==0.10.3.post1
* scikit-learn==1.0.2
* scipy==1.8.0
* numpy==1.18.5
* jupyter==1.0.0
* pandas==1.4.1
* librosa==0.8.0
* yattag==1.14.0
* matplotlib==3.3.3
* Keras==2.7.0

<hr>

# Datasets

* critically endangered Hainan gibbon (Nomascus hainanus) - Link: ... DOI: ...
* critically endangered black-and-white ruffed lemur (Varecia variegata) - Link: ... DOI: ...
* vulnerable Thyolo alethe (Chamaetylas choloensis) - Link: ... DOI: ...
* least concern pin-tailed whydah (Vidua macroura) - Link: ... DOI: ...

# Executing code on Google Colab (easiest way to get started!)

* Thyolo alethe
  - Training (feature extractor frozen - 8 minutes): https://colab.research.google.com/drive/1KZDRrCEjzu3HzP3dLFmDbqXVO0eqEs7V?usp=sharing
  - Predicting (feature extractor frozen): https://colab.research.google.com/drive/1piBHAS5JX8bVdjBXnygejjF9fxmysla8?usp=sharing
  - Training (feature extractor fine-tuned - 8 minutes): https://colab.research.google.com/drive/1o1dmvaOWy6j3hdjAdtZrnV19jRto8Da6?usp=sharing
  - Predicting (feature extractor fine-tuned): https://colab.research.google.com/drive/1xKydHw2aoeVSP_JQtbVFCHni5i5c5pbc?usp=sharing
* Hainan gibbons
  - Training (feature extractor frozen): https://colab.research.google.com/drive/1oYO3UWTL1BfxZn0IA0BYcg8JMGnH980D?usp=sharing
  - Predicting (feature extractor frozen): https://colab.research.google.com/drive/1d-hGRPRIzHubwlTeedqoZCSX8RGp0r4q?usp=sharing
  - Training (feature extractor fine-tuned): https://colab.research.google.com/drive/1uj1OT6JZ7I9z0PRB3YgE11M3ikSaSufa?usp=sharing
  - Predicting (feature extractor fine-tuned): https://colab.research.google.com/drive/1HdnSbvBXyAkYZ4UoA4e-OUpLqeNWm-eV?usp=sharing
* Lemurs
  - Training (feature extractor frozen): https://colab.research.google.com/drive/1Nxa3hDdaR7nKpMdDGA0aBdalxMafRFvw?usp=sharing
  - Predicting (feature extractor frozen): https://colab.research.google.com/drive/1eknVT0M9CTHqLxX29gXi6eITodSJ7Edi?usp=sharing
  - Training (feature extractor fine-tuned): https://colab.research.google.com/drive/1yNv2GctC8B6Z0aIIdlglunctfeR6ZbW7?usp=sharing
  - Predicting (feature extractor fine-tuned): https://colab.research.google.com/drive/1YTn4o8klnFz_Kj9eHwyX4HoA_VVdrk0l?usp=sharing
* Pin-tailed whydah
  - Training (feature extractor frozen): https://colab.research.google.com/drive/1AIhBTQkaUkBrs9ADGU1r4fz1eRGKmyJ9?usp=sharing
  - Predicting (feature extractor frozen): https://colab.research.google.com/drive/1FeLYcKQlE29m64TTRWJ_suuHuI9sxREe?usp=sharing
  - Training (feature extractor fine-tuned): https://colab.research.google.com/drive/1KTOQlnUdY6Jv7wzYympA-PFOUgPqOOsX?usp=sharing
  - Predicting (feature extractor fine-tuned): https://colab.research.google.com/drive/14DkzYCnJXwdSuQno7ONrfX89G9UnvaV8?usp=sharing
