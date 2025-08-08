---
title: 'PhySynthTrainer: A Physics-Based Synthetic Radio Burst Generator and ML Training Toolkit'
tags:
  - radio astronomy
  - machine learning
  - solar physics
  - data simulation
  - synthetic data
  - radio bursts
authors:
  - name: Peijin Zhang
    orcid: 0000-0001-6855-5799
    affiliation: 1
affiliations:
  - name: New Jersey Institute of Technology
    index: 1
date: 2025-08-07
---

# Summary

**PhySynthTrainer** is a lightweight and modular toolkit for generating synthetic solar radio burst data based on physical models and using it to train machine learning models. It supports the simulation of dynamic radio spectra with realistic noise, burst morphologies, and propagation effects. The generated data can be used for training supervised learning models for tasks such as radio burst detection, classification, and real-time space weather monitoring.

The toolkit is intended for researchers and developers working at the intersection of heliophysics, radio astronomy, and machine learning. It allows them to generate controlled, diverse, and label-rich training datasets, bridging the gap between physics-based modeling and data-driven methods.

# Statement of Need

There is a growing interest in applying machine learning to detect and classify solar radio bursts in dynamic spectroscopic data. However, the availability of high-quality, labeled training datasets is limited due to the rarity of events, labeling complexity, and domain noise. 

**PhySynthTrainer** fills this gap by enabling users to simulate a variety of burst types (e.g., type II, III) under different plasma conditions, noise levels, and imaging constraints. It facilitates rapid development and evaluation of ML pipelines and supports reproducibility and benchmarking in radio burst research.

# Software Description

**PhySynthTrainer** is written in Python and designed to be extensible. Key features include:

- A library of physics-based radio burst generators.
- Noise injection and spectral distortion tools.
- Flexible configuration of time-frequency grid resolution.
- Utilities for labeling, exporting, and visualizing synthetic data.
- Example scripts for training basic ML classifiers using generated data.

The software includes built-in unit tests, is fully documented, and can be installed via `pip`.

# Acknowledgements

We thank the community of solar radio observers and machine learning researchers for their support and inspiration. This work was partially supported by the STELLAR project under knowledge dissemination funding.

# References

