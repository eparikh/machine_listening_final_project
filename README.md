# Language-Based Audio Retrieval — DCASE 2024 Task 8

**Emil Parikh, Liam Wintz, Lai Ye**

This repository contains our implementation of modifications to the [Primus SALSA baseline](https://github.com/OptimusPrimus/salsa) for DCASE 2024 Task 8: Language-Based Audio Retrieval. Given a natural language text query, the system retrieves and ranks 10 audio clips from the ClothoV2 dataset by similarity to the query.

Our modifications include a **post-PaSST gated cross-attention aggregation** module and a **Room Impulse Response (RIR) data augmentation** pipeline, both described in the accompanying report.

---

## Repository Structure

```
.
├── experiments/
│   └── ex_dcase24.py          # Main experiment file — model definition, training loop,
│                              # loss, evaluation, etc. Gated cross-attention is
│                              # implemented over here.
│
├── data/
│   ├── data_loader.py
│   └── datasets/
│       └── clotho_v2.py       # ClothoV2 dataset — primary training/eval dataset
│                              # RIR implemented in here
│
├── models/
│   ├── audio/
│   │   ├── passt.py           # PaSST audio encoder wrapper
│   │   ├── base.py            # Splits long audio into 10s windows
│   └── text/
│       └── sentence_embedding_models.py  # RoBERTa-large text encoder wrapper
│
├── scripts/
│   ├── download_clothov2.sh   # Downloads ClothoV2 dataset
│   └── download_checkpoint.sh # Downloads pretrained model checkpoints
│
├── train.s                    # Stage 1 training (contrastive loss)
├── train2.s                   # Stage 2 training (distillation loss)
├── estimate_correspondences.s # generate embeddings for distillation
├── evaluate.s                 # evaluate on ClothoV2 test split
├── environment.yml            # Conda environment specification
└── README.md                  # Oh my god it's this very file. So meta.
```

---

## Our Modifications

### 1. Gated Cross-Attention (`experiments/ex_dcase24.py`)

The key architectural change is in `forward_audio`. The default Primus implementation mean pools the 189 PaSST patch tokens `(B, 189, 768)` into a single vector `(B, 768)`. We replace this with a gated combination of mean pooling and cross-attention:

- The learned `audio_token` parameter `(B, 1, 768)` acts as a query over the 189 patch tokens as keys and values via `nn.MultiheadAttention(768, 8)`
- A scalar gate `g = sigmoid(Linear(768, 1) * mean_pooled + b)` blends the two outputs
- The bias is initialized to `-3.0` so mean pooling dominates early in training (`g ≈ 0.047`)
- Activated via config: `audio_features.aggregate=weighted_single`

### 2. RIR Data Augmentation (`data/datasets/wrapper.py`)

An augmentation class that convolves ClothoV2 clips with random Room Impulse Responses from the [OpenSLR SLR28](https://www.openslr.org/28/) dataset (60,000 simulated small/medium/large room RIRs). During training the augmented dataset is concatenated with the original ClothoV2, doubling the training set size.

---

## References

- P. Primus and G. Widmer, "A Knowledge Distillation Approach to Improving Language-Based Audio Retrieval Models," DCASE 2024 Technical Report.
- K. Drossos, S. Lipping, and T. Virtanen, "Clotho: An Audio Captioning Dataset," ICASSP 2020.
