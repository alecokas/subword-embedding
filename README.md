# subword-embedding
A tool for generating sub-word (phone or grapheme) level embeddings from an HTK-style MLF ASR corpus as used in https://github.com/alecokas/BiLatticeRNN-Confidence

The ground truth transcription from the audio recording (*.mlf) is required to build the corpus. Additionally a summary file mapping subword units to the respective phonetic pronounciation or Latin representation is required. An example snippet for the Georgian language is provided below:

```
ა G1;D1 GEORGIAN LETTER AN
ვ G2;D1 GEORGIAN LETTER VIN
ს G3;D1 GEORGIAN LETTER SAN
...
```

### Training
Run the following command:
```
python embed_subwords.py [arguments]
```

### Dependencies
* python 3.6.3
* numpy 1.14.0
* matplotlib 2.1.2
* scikit-learn 0.19.1
