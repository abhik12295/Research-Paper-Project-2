# Active Synthetic Data Generation (ASDG) using Conditional GANs

### A Lightweight Generative AI Framework for Scarce-Label Image Classification

<p align="center">
<b>Bridging Generative AI, Active Learning, and Data Efficiency for Real-World AI Systems</b>
</p>


## 📌 Abstract

Deep neural networks depend heavily on large labeled datasets, limiting their applicability in real-world, data-scarce environments.
This project presents a **lightweight Active Synthetic Data Generation (ASDG) framework**, integrating:

* **Conditional Generative Adversarial Networks (cGANs)**
* **Uncertainty + Diversity-based Active Learning**

The framework generates class-conditioned synthetic samples and selectively incorporates them into training pipelines to improve classification performance under limited supervision.

📊 Experimental results on **CIFAR-10 (300 samples/class)** show:

* Baseline: **60.98%**
* Random Synthetic Augmentation: **67.73%**
* Active Synthetic Augmentation: **68.69% (best)**


## 🎯 Research Goal
To answer a critical research question:

> **Can actively selecting synthetic data improve learning efficiency compared to random augmentation in scarce-label settings?**

## 🧠 Key Contributions
* ✅ Novel integration of **cGAN + Active Learning**
* ✅ Lightweight, reproducible framework for **resource-constrained environments**
* ✅ Demonstration of **synthetic data effectiveness under limited labels**
* ✅ Empirical analysis of **informativeness vs stability trade-offs**
* ✅ Practical pipeline applicable to real-world domains



## ⚙️ System Workflow

```text
Few Labeled Data
        │
        ▼
Conditional GAN (cGAN)
        │
        ▼
Synthetic Data Pool
        │
        ├── Random Sampling
        └── Active Selection (Uncertainty + Diversity)
                         │
                         ▼
                 Student Model (CNN)
                         │
                         ▼
               Performance Evaluation
```


## 🔬 Methodology

### 1️⃣ Conditional GAN (cGAN)

Synthetic samples are generated using:

```math
x_{syn} = G(z, y)
```

* `z` → latent noise vector
* `y` → class label
* Enables **class-conditioned image generation**


### 2️⃣ Active Sample Selection

Each synthetic sample is scored using:

```math
S(x) = U(x) + \lambda D(x)
```

Where:

* **U(x)** → entropy-based uncertainty
* **D(x)** → feature diversity
* **λ** → diversity weighting factor

👉 Ensures selection of **informative + non-redundant samples**



### 3️⃣ Iterative Training Pipeline

* Train baseline model on real data
* Generate synthetic pool using cGAN
* Select samples:

  * Random OR Active
* Retrain student model
* Repeat for multiple rounds



## 📊 Experimental Setup

* Dataset: **CIFAR-10**
* Training samples: **300 per class (3000 total)**
* Synthetic samples: **400 per round**
* Rounds: **3 iterations**
* Hardware: Apple Silicon (MPS backend)


## 📈 Results

| Method                  | Accuracy (%) |
| ----------------------- | ------------ |
| Real-only baseline      | 60.98        |
| Random synthetic        | 67.73        |
| Active synthetic (ASDG) | **68.69**    |

---

## 📌 Key Insights

* Synthetic data significantly improves performance in low-data regimes
* Active selection enhances **sample efficiency and peak accuracy**
* Trade-off observed:

  * Higher informativeness
  * Reduced training stability

👉 Highlights a key research direction:
**Balancing informativeness with robustness in synthetic data pipelines**

## Installation

```bash
git clone https://github.com/abhik12295/Research-Paper-Project-2.git
cd Research-Paper-Project-2

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```


## Usage

```bash
# Stage 1: Train CGAN
python stage1_cgan_cifar10.py

# Stage 2: Baseline Model
python stage2_student_baseline.py

# Stage 3: Active Synthetic Learning
python stage3_asdg_cgan.py
```


## 🌍 Real-World Applications

This framework is highly relevant for:

* 🏥 Healthcare (limited labeled imaging data)
* 🚚 Logistics & supply chain AI
* 🔐 Security & surveillance systems
* 🛰️ Remote sensing / satellite imagery
* 🤖 Autonomous systems


## 🔬 Research Significance

This work contributes to **data-efficient AI systems**, demonstrating that:

* Generative models can **replace expensive labeling pipelines**
* Active learning can **maximize information gain per sample**
* Lightweight architectures can still achieve **strong performance gains**


## 🚀 Portfolio & Immigration Impact (EB1/O1 Alignment)

This project demonstrates:

* 📌 Original contribution in **Generative AI + Active Learning**
* 📌 Applied research with measurable performance improvements
* 📌 Implementation of **state-of-the-art deep learning techniques**
* 📌 Relevance to **industry-scale AI challenges**

## 📚 References

Based on foundational works in:

* GANs (Goodfellow et al.)
* Conditional GANs (Mirza & Osindero)
* Active Learning (Settles, Ash et al.)

## 👨‍💻 Author

**Abhishek Kumar**
PhD (AI), University of the Cumberlands
Specialization: Generative AI, Data Engineering, Predictive Systems


## 🔮 Future Work

* Diffusion Models for higher-quality synthesis
* Stability-aware training mechanisms
* Adaptive active learning strategies
* Large-scale dataset generalization


## ⭐ Acknowledgment

If you find this work valuable, consider ⭐ starring the repository.
