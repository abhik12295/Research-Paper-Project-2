🚀 Active Synthetic Data Generation (ASDG) using Conditional GANs
AI-Driven Data Augmentation for Scarce-Label Image Classification
<p align="center"> <b>Bridging Generative AI + Active Learning to solve real-world data scarcity</b> </p>

Project Highlights
- Advanced Generative AI: Conditional GAN (CGAN) for class-aware image synthesis
- Active Learning Integration: Intelligent sample selection improves model efficiency
- End-to-End ML Pipeline: From data generation → training → evaluation
- Research-Grade Implementation: Designed for academic publication & experimentation
- Demonstrated Performance Gains over baseline models


Problem Statement

Modern deep learning models require large labeled datasets, which are often:

Expensive to annotate
Time-consuming to curate
Impractical in niche domains

This project addresses the challenge by combining:

Synthetic data generation (GANs)
Data efficiency techniques (Active Learning)

👉 Goal: Maximize model performance with minimal labeled data


            +----------------------+
            |  Real Dataset (Small)|
            +----------+-----------+
                       |
                       v
        +---------------------------+
        | Conditional GAN (CGAN)    |
        | - Learns data distribution|
        | - Generates labeled data  |
        +------------+--------------+
                     |
                     v
        +---------------------------+
        | Active Learning Engine    |
        | - Uncertainty Sampling    |
        | - Informative selection   |
        +------------+--------------+
                     |
                     v
        +---------------------------+
        | Student Model (CNN)       |
        | - Trained iteratively     |
        +------------+--------------+
                     |
                     v
        +---------------------------+
        | Performance Evaluation    |
        | - Accuracy Improvement    |
        | - Baseline Comparison     |
        +---------------------------+