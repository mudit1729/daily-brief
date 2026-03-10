# Slide Trace Matrix Summary

This summary accompanies `Slide_Trace_Matrix.csv` and gives per-lecture status counts from the current slide-to-summary matching pass.

| Lecture | PDF | Summary | Slides | High | Medium | Duplicate | Recap | Admin | Visual/Review | Needs Manual Review |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `lec1.intro.pdf` | `Lec01_Introduction_to_Deep_Learning.md` | 110 | 18 | 33 | 9 | 5 | 6 | 0 | 39 |
| 2 | `lec2.universal.pdf` | `Lec02_Universal_Approximation.md` | 163 | 24 | 43 | 44 | 9 | 6 | 0 | 37 |
| 3 | `lec3.learning.pdf` | `Lec03_Learning_Backpropagation.md` | 135 | 12 | 40 | 34 | 6 | 4 | 0 | 39 |
| 4 | `lec4.learning.presented.pdf` | `Lec04_Learning_Continued.md` | 101 | 1 | 32 | 18 | 3 | 6 | 0 | 41 |
| 5 | `lec5.pdf` | `Lec05_MLP_and_Backprop.md` | 175 | 16 | 45 | 67 | 3 | 5 | 0 | 39 |
| 6 | `lec6.pdf` | `Lec06_Optimization_Basics.md` | 154 | 30 | 44 | 45 | 10 | 6 | 0 | 19 |
| 7 | `lec7.stochastic_gradient.pdf` | `Lec07_Stochastic_Gradient_Descent.md` | 122 | 20 | 34 | 42 | 5 | 5 | 0 | 16 |
| 8 | `lec8.optimizersandregularizers.pdf` | `Lec08_Optimizers_and_Regularizers.md` | 143 | 25 | 32 | 37 | 7 | 5 | 2 | 35 |
| 9 | `Lec9.CNN1.pdf` | `Lec09_CNN_Part1.md` | 288 | 28 | 92 | 126 | 5 | 9 | 0 | 28 |
| 10 | `lec10.CNN2.pdf` | `Lec10_CNN_Part2.md` | 158 | 47 | 35 | 43 | 4 | 7 | 1 | 21 |
| 11 | `Lec11.CNN3.pdf` | `Lec11_CNN_Part3.md` | 224 | 20 | 34 | 119 | 17 | 9 | 0 | 25 |
| 12 | `Lec12.CNN4.pdf` | `Lec12_CNN_Part4.md` | 117 | 31 | 26 | 19 | 7 | 9 | 1 | 24 |
| 13 | `lec13.recurrent.pdf` | `Lec13_Recurrent_Networks_Part1.md` | 132 | 18 | 31 | 30 | 3 | 9 | 0 | 41 |
| 14 | `lec14.recurrent.pdf` | `Lec14_Recurrent_Networks_Part2.md` | 127 | 17 | 39 | 31 | 7 | 9 | 0 | 24 |
| 15 | `lec15.recurrent.pdf` | `Lec15_Recurrent_Networks_Part3.md` | 92 | 18 | 22 | 18 | 5 | 9 | 0 | 20 |
| 16 | `lec16.recurrent.pdf` | `Lec16_Recurrent_Networks_Part4.md` | 101 | 14 | 32 | 17 | 2 | 9 | 0 | 27 |
| 17 | `lec17.recurrent.pdf` | `Lec17_Recurrent_Networks_Part5.md` | 174 | 24 | 51 | 53 | 7 | 11 | 1 | 27 |
| 18 | `lec18.attention.pdf` | `Lec18_Attention_Mechanisms.md` | 187 | 12 | 50 | 59 | 2 | 9 | 0 | 55 |
| 19 | `lec19.Txfmr_GCN.pdf` | `Lec19_Transformers_and_GCN.md` | 97 | 8 | 23 | 33 | 5 | 1 | 0 | 27 |
| 20 | `lec20.representations.pdf` | `Lec20_Representation_Learning.md` | 101 | 18 | 25 | 21 | 2 | 9 | 0 | 26 |
| 21 | `lec21.VAE.pdf` | `Lec21_Variational_Autoencoders.md` | 66 | 16 | 18 | 14 | 1 | 9 | 0 | 8 |
| 22 | `lec_22_GAN1.pdf` | `Lec22_GANs_Part1.md` | 46 | 8 | 19 | 8 | 0 | 1 | 0 | 10 |
| 23 | `Gans_TA.pdf` | `Lec23_GANs_Part2.md` | 53 | 10 | 8 | 7 | 1 | 1 | 2 | 24 |
| 24 | `lec26.hopfield.pdf` | `Lec24_Hopfield_Networks.md` | 126 | 17 | 65 | 16 | 6 | 1 | 1 | 20 |
| 25 | `lec27.BM.pdf` | `Lec25_Boltzmann_Machines.md` | 108 | 27 | 41 | 11 | 8 | 1 | 0 | 20 |

## Status Meanings

- `covered_high_confidence`: strong automatic match between slide text and a specific summary section
- `covered_medium_confidence`: plausible automatic match, but should still be spot-checked if strict proof is required
- `duplicate_build`: slide appears to be an animation/build of the previous slide and inherits its coverage target
- `recap_reference`: recap slide that usually points to concepts already covered elsewhere in the same summary
- `admin_title` / `admin_poll`: opening, agenda, or poll slides
- `visual_only_or_needs_review`: too little extractable text to classify confidently
- `needs_manual_review`: content slide with weak automatic alignment to a summary section
