# Code for the Paper  
**“On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples”**

This repository contains the code accompanying the experiments described in the paper.  
Each section of the paper is linked to a corresponding folder in this repository.

---

## 📦 Required Packages

- **numpy**  
- **scipy**  
- **pandas**  
- **scikit-learn**  
- **pytorch**  
- **tick** (for files in `numerical_validations/`)  
- **tsai** (for files in `retinal_ganglion_cell_stimulus_classification/`)  
- **spikebench** (for files in `retinal_ganglion_cell_stimulus_classification/`)

---

## 🔍 Section 3 & Appendix B  
Code is located in:

```
numerical_validations/
```

---

## 🔍 Section 4.1 — Synthetic Data Experiment  
Files can be found in:

```
a_synthetic_data_experiment/
```

---

## 🔍 Section 4.2 — Retinal Ganglion Cell Stimulus Classification  

Files are located in:

```
retinal_ganglion_cell_stimulus_classification/
```

This experiment requires the following external packages:

- **tsai**  
  👉 https://github.com/timeseriesAI/tsai  
- **spikebench**  
  👉 https://github.com/lzrvch/spikebench  

### Example command

```bash
python3 main_retina.py   --data-type retina12   --features isi   --features2 isi_sdfa0c isi_sdfa1c   --model-type FCNPlus   --use-log   --gpu 0   --seeds 0   --n_epochs 200   --batch_size 256   --lr 0.1   --algorithm SGD   --scheduler flatcosine   --use-CNN-for-additional   --dataseed 0
```

---

## 🔍 Section 4.3 — Human Neural Spike Train Analysis  

Files are located in:

```
an_analysis_of_human_neural_spike_trains/
```

---

## 🔍 Section 4.4 — Amino Acid Contact Analysis  

Files are located in:

```
an_analysis_of_amino_acid_contacts/
```

The Riemannian geometric manifold learning algorithm used in Section 4.4  
is implemented in the following file:

```
an_analysis_of_amino_acid_contacts/
  Riemannian_distortion_minimization/
    main_amino_acid.m
```

---

If you use this code, please cite the accompanying paper.
