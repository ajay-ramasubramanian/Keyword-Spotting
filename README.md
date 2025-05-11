# Keyword Spotting using discret audio feautres

## Project Overview

Keyword Spotting is one amoung many Speech Recognition Task that has been deeply researched topics. In our project I will be focusing on using **Discrete Audio Representation for performing Keyword Spotting task.**

You may wonder why make use of *discrete audio representation* when we could get staggering performance from using *continueous audio representation* for Keyword Spotting task?

The reason being **discrete audio representation** are representation that involve audio features that are inherently discrete(i.e) They have a limited codebook to capture relevant Information in the audio waveform.

This differentiates them from traditional features like **FBANKs or MFCCs**, as well as self-supervised features such as **Wav2Vec, Hubert, and WavLM**, which are continuous in nature. The appeal of discrete audio representations lies in their potential benefits, such as improved integration with large multimodal language models and NLP pipelines. Additionally, they transform regression problems into classification ones, which are generally easier to handle

**CHALLENGES**:

- ***Limited Information Capture***: Discrete audio representations typically use a limited codebook to capture information from audio waveforms, which potentially leads to loss of detailed information compared to continuous representations.

- ***Vocabulary Size and Generalization***: Managing a limited vocabulary size in discrete representations poses challenges in accurately representing diverse audio content, which                                               affects the model's ability to generalize across different speech patterns and accents.

- ***Complexity of Modeling***: Discrete representations often require more complex models to effectively capture and utilize the discrete features, as opposed to continuous                                         representations which can be processed more straightforwardly using conventional techniques like convolutional and recurrent neural networks.

- ***Integration with Existing Systems***: Integrating discrete representations into existing speech processing systems or frameworks may require significant modifications and                                                   adaptations, as they may not directly align with the pipelines and architectures designed for continuous audio representations.

- ***Performance Gap***: Despite advancements, discrete representations still generally lag behind continuous representations in terms of performance metrics such as accuracy and                            robustness, posing a challenge for their widespread adoption in real-world applications. Closing this performance gap remains a key challenge in the field.

## Dataset

For this project I will be making use of ***Google Speech Commmand Dataset***. Here is a breif description of the Dataset:

An audio dataset of spoken words designed to help train and evaluate keyword spotting systems. Its primary goal is to provide a way to build and test small models that detect when a single word is spoken, from a set of ten target words, with as few false positives as possible from background noise or unrelated speech. <br>Note that in the train and validation set, the label "unknown" is much more prevalent than the labels of the target words or background noise. One difference from the release version is the handling of silent segments. <br>While in the test set the silence segments are regular 1 second files, in the training they are provided as long segments under "background_noise" folder.<br> Here we split these background noise into 1 second clips, and also keep one of the files for the validation set.

## Goal of the project

The Goal of the Project is to evaluate the performance of discrete audio representations for keyword spotting.

To do that I will be , designing a system that works well with discrete audio features. I will be considering **EncoDec** features which are implemented in **speechbrain**. On top of that, I will device an architecture that achieves the best performance.

I will be comparing different types of neural networks, such as RNNs, Transformers, CNNs, MLPs, etc. Finally the performance of the system using discrete audio representation is compared with that achieved by standard features and by popular self-supervised models such as **Wav2vec, Hubert, and WavLM**.  

## Key Components

Continuous Audio Representations:
- WavLM
- HuBERT
-  Wav2vec

### Discrete Audio Representations:
- EnCodec
-  Discrete Autoencoder Codec (DAC)

### Neural Architectures:

- Feed-forward (XVector)
- Recurrent Networks (RNN, LSTM, GRU)
- Transformers(Conformer, Transformer)
- CRDNN

## Technical Details:

  - Loss function: Negative Log Likelihood

  - Primary task: Keyword Spotting

  - Technology stack: Python, PyTorch, SpeechBrain, Matplotlib

## Results Summary

All models underwent rigorous hyperparameter optimization. We tested multiple neural architectures with RNNs, LSTMs, and Transformers showing particular promise.
Continuous Features Performance

#### Continuous representations contain fine-grained audio information and performed exceptionally well:

  - Fbanks: Achieved 2% test error rate without overfitting

  - Wav2Vec: Reached 1.67% error rate despite slight overfitting due to fine-tuning on limited data

  - HuBERT: Attained 1.46% test error rate with similar overfitting characteristics

  - WavLM: Best performer at 0.92% error rate with rapid convergence

#### Discrete Features Performance

Discrete tokens offer compact, categorical audio representations:

  - EnCodec: Reached 7.6% test error rate without overfitting, showing promise with sequence-based models

## Conclusion

Continuous feature models deliver superior accuracy (as low as 0.92% error with WavLM) but require more computational resources. Discrete feature models achieve reasonable performance (7.6% error rates) while offering greater efficiency. This represents a clear trade-off between accuracy and computational demands in audio processing systems.
