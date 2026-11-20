# Language, Culture and Editorial Differences in Historical isiXhosa Newspapers: A Comparative NLP Study

Code for the Project 2 paper (Natural Language Processing, *Language in Culture and Responsible NLP*), University of the Witwatersrand.

**Authors:** Lulama Hlungwani, Naledi Magomotsi, Ntokozo Skosana 

---

## Overview

Historical newspapers record African-language public life, but they are hard to study computationally. isiXhosa and English share pages, spelling varies, advertisements are common, and text is transcribed by page rather than by article.

This project compares two 1880s isiXhosa newspapers in the years both were published:

- ***Imvo Zabantsundu*** (IMVO): founded by J. T. Jabavu in King William's Town in 1884, independent of missionary control
- ***Isigidimi sama-Xosa*** (ISIG): published at Lovedale, one of the earliest African-language newspapers

**Research question:** How do IMVO and ISIG differ in language and topics, and what can responsible NLP legitimately infer from those differences?

The paper does **not** assume that a newspaper's identity reveals its editorial or political views. It pairs topic modelling with classification and shortcut checks to test what the models are actually picking up.

---

## Data

Transcribed pages from the isiXhosa Intellectual Traditions digital archive (Schoots et al., 2023).

| Step | Count |
|---|---|
| Full archive | 3,608 pages, 718 issues |
| After removing 2 exact duplicate pages and keeping the shared years (1884, 1885, 1887, 1888) | **1,089 pages** (633 IMVO, 456 ISIG), 216 issues |
| Text sections (400 words, minimum 200 words; 6 pages too short) | used for topic modelling |



Preprocessing: text was standardised and the full newspaper-name phrases were removed. Because paragraph breaks usually reflect page layout rather than article boundaries, pages were split into 400-word sections.

---

## Methods

### Topic modelling (NMF)

- Non-negative matrix factorisation on 400-word sections
- Tested 4, 6, 8 and 10 topics on a balanced sample of 902 sections. Reconstruction error changed only slightly (28.19 → 27.60), so **8 topics** were chosen for interpretability
- Topic weights averaged per page; **324 pages per newspaper**, matched by year, compared

### Page-level classification

- Majority-class baseline
- **Word-level** and **character-level** TF-IDF + logistic regression (balanced class weights; tuned over n-gram range, minimum document frequency and C ∈ {0.1, 1, 10})
- **Issue-level splits** so pages from the same issue never appear in different sets: train 650 / validation 215 / test 224 pages (44 test issues), balanced by year and newspaper
- Models selected on validation macro-F1, then retrained on train + validation before testing

### Shortcut baselines

Depth-2 decision trees using:

1. Page length alone
2. Page length + English function-word rate

### Post-hoc robustness checks

Selected newspaper-name and date terms masked, and evaluation on non-front pages only. These were run **after** inspecting the main test results, so they are not independent confirmation.

---

## Key Results

**Topics (mean weight, IMVO vs ISIG)**

| Topic | IMVO | ISIG |
|---|---|---|
| General isiXhosa | 0.13 | 0.60 |
| English | 0.24 | 0.02 |
| Advertising-related (5 topics combined) | 0.38 | 0.09 |
| Politics / institutions | 0.25 | 0.30 |

**Classification (held-out test set, 224 pages)**

| Model | Accuracy | Balanced acc. | Macro-F1 | Val. macro-F1 |
|---|---|---|---|---|
| Majority baseline | 0.571 | 0.500 | 0.364 | 0.371 |
| Word TF-IDF | 1.000 | 1.000 | 1.000 | 0.976 |
| Character TF-IDF | 1.000 | 1.000 | 1.000 | 0.990 |
| Page length only (tree) | n/a | n/a | 0.788 | n/a |
| Page length + English-word rate (tree) | n/a | n/a | 0.878 | n/a |

**Main takeaway:** the newspapers are easy to tell apart, but mainly through **language mix and page structure**, not demonstrated editorial views. Simple baselines already reach macro-F1 0.79–0.88, and the top word features included dates, newspaper names, place names and common English words.

---

## Limitations

- The test set has only 44 issues (12 ISIG), so perfect scores carry wide uncertainty and no confidence intervals are reported
- Nearby issues and repeated advertisements may still leak across the issue-level split
- Masking and non-front-page checks were run after seeing test results, and masking removes only selected terms
- The papers differ in format (four vs eight pages per issue), and IMVO covers only November–December 1884
- Topic results are descriptive, from a single fit with no uncertainty estimates, and need validation by isiXhosa readers
- Transcription quality and missing article boundaries may affect results

## Responsible Use

We would **not** deploy this kind of classifier as an ideology or authorship detector. The archive reflects what editors, missionaries and transcribers chose to preserve, so results should not be read as knowledge about a whole community. Future work should hold out whole years, add issue-level uncertainty, and pair topic analysis with close reading by isiXhosa speakers.

---

## Repository Structure

<!-- TODO: Replace with your actual layout. -->

```
.
├── data/                  # (or instructions for obtaining the archive)
├── <preprocessing>.py     # Cleaning, deduplication, 400-word sectioning
├── <topic_model>.py       # NMF topic modelling
├── <classification>.py    # TF-IDF classifiers, issue-level splits, baselines
├── <figures/results>/     # Figures and tables used in the paper
└── README.md
```

## Getting Started

<!-- TODO: Fill in with your real environment and commands. -->

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt   # likely scikit-learn, numpy, pandas, matplotlib

# 1. Preprocess the data
python <preprocessing>.py

# 2. Run NMF topic modelling
python <topic_model>.py

# 3. Train and evaluate classifiers and shortcut baselines
python <classification>.py
```

**Compute:** hardware, memory and run times have not been documented yet. Add them here if you can.

---

## Project Context and Contributions

This repository comes from a three-person group project. Each member took a main area, with literature review, experiment planning and report review shared:

- **Naledi Magomotsi:** data preparation and augmentation
- **Ntokozo Skosana:** embeddings and language change
- **Lulama Hlungwani:** model evaluation and responsible NLP


## References

- Schoots, J., Ngwendu, A., De Wet, J., & Muftic, S. (2023). *IsiXhosa intellectual traditions digital archive: Digitizing isiXhosa texts from 1870–1914.* RAIL 2023, pp. 54–64.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, 401, 788–791.
- Hill, M. J., & Hengchen, S. (2019). Quantifying the impact of dirty OCR on historical text analysis. *Digital Scholarship in the Humanities*, 34(4), 825–843.
- Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.
- Blodgett, S. L., et al. (2020). Language (technology) is power: A critical survey of "bias" in NLP. *ACL 2020*, pp. 5454–5476.



## License


Lulama Unity Hlungwani, Naledi Magomotsi and Ntokozo Skosana
