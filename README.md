# 🏥 Medical ASR — Domain Adaptation Study
### Whisper LoRA vs DoRA vs Wav2Vec2 on Medical Speech

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://www.kaggle.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.12-green?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Overview

This project investigates **parameter-efficient fine-tuning** of speech recognition models for the medical domain. We take OpenAI's **Whisper-small** (pre-trained on 680K hours of general speech) and adapt it to medical speech using two PEFT methods — **LoRA** and **DoRA** — then benchmark both against a fully fine-tuned **Wav2Vec2-base** model.

The key question: *can adapting just ~1% of parameters match or beat full fine-tuning on domain-specific speech?*

---

## 🏆 Results

> Evaluated on a held-out 200-sample subset of the [Medical Speech, Transcription & Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent) dataset.

| Model | Method | Trainable Params | WER ↓ | CER ↓ | vs Baseline |
|-------|--------|-----------------|-------|-------|-------------|
| Whisper-small | No fine-tuning (baseline) | 244M (100%) | 17.93% | 7.78% | — |
| **Whisper-small** | **LoRA (r=16, use_dora=False)** | **2.3M (~1%)** | **2.81%** | **1.64%** | **▼ 84.3%** |
| Whisper-small | DoRA (r=16, use_dora=True) | 2.3M (~1%) | 9.50% | 4.52% | ▼ 47.0% |
| Wav2Vec2-base | Full fine-tuning (CNN frozen) | ~90M | 14.61% | 8.42% | N/A |

**LoRA wins decisively** — with only 1% of trainable parameters, it achieved an **84.3% reduction in Word Error Rate** over the untuned baseline, outperforming both DoRA and full Wav2Vec2 fine-tuning.

---

## 📊 Training & Results Visualization

These plots illustrate training dynamics and qualitative performance across models.

### 🔹 Example Predictions (LoRA)
![Predictions](./images/predictions.png)

### 🔹 LoRA Training Curve
![LoRA](./images/LORA_training_loss.png)

### 🔹 DoRA Training Curve
![DoRA](./images/dora_training.png)

### 🔹 Wav2Vec2 Training Curve
![Wav2Vec2](./images/wav2vec2_training.png)



## 🗂️ Project Structure

```
medical-asr-domain-adaptation/
│
├── notebook.ipynb           # Full experiment — training, eval, inference
├── final_results.csv        # WER / CER comparison table
├── README.md
│
└── images/
    ├── LORA.png
    └── LORA_training_loss.png
```

---

## 🧠 Methodology

### Dataset
- **Medical Speech, Transcription & Intent** (Kaggle) — ~6,600 audio clips of medical symptom descriptions
- 80 / 10 / 10 train / val / test split (`random_state=42`)
- Evaluation on a fixed 200-sample subset for reproducibility

### Models

| Model | Description |
|-------|-------------|
| `openai/whisper-small` | 244M parameter encoder-decoder ASR model, pre-trained on 680K hrs |
| `facebook/wav2vec2-base` | 95M parameter self-supervised model, fully fine-tuned with CTC head |

### PEFT Configuration (LoRA / DoRA)
```python
LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout   = 0.05,
    use_dora       = False,   # True for DoRA
)
```

### Training Setup
- **Hardware**: Kaggle Tesla T4 (15.6 GB VRAM)
- **Whisper LoRA/DoRA**: Seq2SeqTrainer, fp16, 3 epochs
- **Wav2Vec2**: Trainer with CTC loss, CNN layers frozen

---

## 🚀 Quick Start

### Load the LoRA adapter (inference)
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
import librosa, torch

model_name = "openai/whisper-small"
processor  = WhisperProcessor.from_pretrained(model_name)
base       = WhisperForConditionalGeneration.from_pretrained(model_name)
model      = PeftModel.from_pretrained(base, "your-username/whisper-small-medical-lora")
model.eval()

def transcribe(audio_path):
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs   = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs, language="english", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0]
```

### Reproduce the full experiment
1. Open the [Kaggle notebook](https://www.kaggle.com/code/aymendhieb1/speechtotext-whisper-finetunning-with-medspeech)
2. Attach the Medical Speech dataset + the 3 model datasets
3. Run all cells — no GPU needed for evaluation only



## 💡 Key Takeaways

- **LoRA at 1% parameters beats full fine-tuning** on this domain adaptation task
- **DoRA underperformed LoRA** here despite its theoretical advantages — likely needs more epochs or a higher rank
- **Wav2Vec2 full fine-tuning** still couldn't beat LoRA-adapted Whisper, showing the power of starting from a strong pre-trained ASR base
- Medical speech is a niche domain where even small adaptation gains matter significantly for real-world usability

---

## 📚 References

- [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Radford et al., OpenAI
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al.
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) — Liu et al.
- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) — Baevski et al., Meta AI

---


