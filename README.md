Language, Culture and Editorial Differences in Historical isiXhosa Newspapers
COMS4054A - Project 2 (code README)

WHAT THIS CODE DOES
-------------------
Compares Imvo Zabantsundu (IMVO) and Isigidimi sama-Xosa (ISIG) in the four
years both newspapers appear (1884, 1885, 1887, 1888):
  * NMF topic modelling on 400-word windows
  * page-level newspaper classification (word / character TF-IDF + logistic
    regression), evaluated with whole issues held out of training
  * simple shortcut baselines (page length, English-word rate)

FOLDER LAYOUT
-------------
  config.py               shared paths/settings (imported by every script)
  preprocessing.py        step 1
  modelling.py            step 2
  classification.py       step 3
  derived_statistics.py   step 4
  make_figures.py         step 5 (the two figures used in the abstract)
  run_all.py              runs steps 1-5 in order
  requirements.txt        Python packages
  data/                   PUT THE DATASET ZIP HERE (not included)
  outputs/                created automatically; all results are written here

HOW TO RUN (VS Code)
--------------------
1. Install Python 3.10 or newer and open this folder in VS Code.
2. Create a virtual environment and install the packages:
       python -m venv .venv
       .venv\Scripts\activate          (Windows)   or   source .venv/bin/activate   (macOS/Linux)
       pip install -r requirements.txt
3. Copy the dataset zip (named like "1. TxtFiles1870-1894_TB.zip") into the
   data/ folder. Any zip whose name contains "TxtFiles1870-1894" is found.
4. Run everything:
       python run_all.py
   or run the scripts one at a time, IN THIS ORDER (each one needs the files
   written by the one before it):
       python preprocessing.py
       python modelling.py
       python classification.py
       python derived_statistics.py
       python make_figures.py
   (In VS Code you can also open a script and press the Run button.)

Figures are saved to outputs/figures/ and no plot windows open. To also see
them on screen, set SHOW_PLOTS=1 first:
       set SHOW_PLOTS=1                 (Windows cmd)
       $env:SHOW_PLOTS=1                (Windows PowerShell)
       export SHOW_PLOTS=1              (macOS/Linux)

RUNTIME
-------
Roughly 5-10 minutes on a normal laptop (most of it is classification.py, which
fits 24 TF-IDF/logistic-regression configurations).

KEY OUTPUT FILES (outputs/)
---------------------------
  classification_final_test_results.csv     Table 1 (test metrics)
  classification_validation_comparison.csv  validation macro-F1 used for selection
  word_/character_model_validation_results.csv   full tuning grids
  shortcut_baselines.csv / .png             shortcut baselines vs TF-IDF models
  nmf_year_matched_topic_summary.csv        Figure 1 values
  nmf_topics_8.csv                          top words of the 8 topics
  derived_statistics.csv, validation_selection_summary.csv   numbers quoted in the abstract
  figures/paper_fig1_topics.png, paper_fig2_shortcuts.png    figures in the abstract

REPRODUCIBILITY
---------------
All random choices use RANDOM_STATE = 42 (config.py). Exact numbers can differ
slightly if different versions of scikit-learn/pandas are installed.
The topic NAMES (e.g. "Retail (soap, shops)") are our own interpretation of
each topic's top words and passages, not output of the code.

NOTE ON POST-HOC ANALYSES
-------------------------
The identity-masking and non-front-page checks in classification.py were run
after the main test results had been inspected; they are sensitivity analyses,
not independent confirmatory tests.

AUTHORS
Lulama Unity Hlungwani, Naledi Magomotsi and Ntokozo Skosana
