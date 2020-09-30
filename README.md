## EMBODY Study Code
##### EMBODY Framework: Evaluating Multivariate Maps of Body Awareness to measure internal attention states during meditation.
*A collection of iPython Notebook and MATLAB scripts used to pre-process and analyse EMBODY data.*

------------------
### System Requirements

- [Python 2.7](https://www.python.org/downloads/)
    -  Python dependencies: 
        - [numpy](https://www.scipy.org/install.html)
        - [pandas](https://www.scipy.org/install.html)
        - [scipy](https://www.scipy.org/install.html)
- [Jupyter Notebooks](https://jupyter.org/install)
- [MATLAB](https://www.mathworks.com/products/get-matlab.html)
- [Princeton MVPA Toolbox](https://pni.princeton.edu/pni-software-tools/mvpa-toolbox)
    - Add three additional Princeton MVPA Toolbox files from the **mvpa_code/logreg** folder to **./princeton-mvpa-toolbox-master/core/learn**
        - *classifierLogisticRegression.m*
        - *test_L2_RLR.m*
        - *train_L2_RLR.m*
- [Consented fMRI data](https://ucsf.box.com/s/l4g4nj7ryy6s48p6dtc0c9qraf04za3y)
    - This is the fMRI data to run the MVPA scripts with. 
    - Directory: embody_fmri_consented
- [Consented MVPA output data](https://ucsf.box.com/s/l4g4nj7ryy6s48p6dtc0c9qraf04za3y)
    - This contains the full MVPA results output for Step 1 and Step 2 (may need this to run demo with 124, files too large for Github)
    - Also includes spreadsheet of participant demographics and attention metrics
    - Directory: embody_mvpa_output_consented
    
-------------------
### Installation Guide

1. Clone or download repository. 
2. View and run scripts in Jupyter Notebook (ipynb) or MATLAB (.m) depending on script type.

-------------------
### Description

EMBODY Framework: Evaluating Multivariate Maps of Body Awareness to measure internal attention states during meditation.

> *Step 1*. Brain pattern classifier training. Machine learning algorithms are trained in fMRI neural patterns associated with internal mental states in the Internal Attention (IA) task. IA is directed via auditory instructions to pay attention with eyes closed to the breath, mind wandering, self-referential processing, and control conditions of attention to the feet and ambient sounds (see Fig. 2). Unique individualized brain patterns for each participant are learned using n-1 cross-validation with 6 blocks of the IA task.

> *Step 2*. Meditation period classification. Neural patterns are collected during a 10-min meditation period (in this case, focused attention to the breath; administered in the middle of 6 IA blocks), and are decoded by multi-voxel pattern analysis (MVPA) using the unique brain patterns learned in Step 1. Meditation is characterized second-by-second into mental states of attention to breath (B), mind wandering (MW), or self-referential processing (S), producing a read-out of distinct and fluctuating mental states during meditation.

> *Step 3*. Quantification of internal attention during meditation. From the temporal read-out of meditative mental states in Step 2, novel attention metrics during meditation can be quantified including percentage time spent in each mental state, number of times engaged in each mental state (“events”), and mean duration spent in each mental state. See Methods for details.

-------------------

### Instructions for use

**1. Machine Learning, Step1 & Step2**
- *batch_embody_step1.m*
    - Define: 
        - path to EMBODY root directory,
        - subject IDs (strings, e.g. '101'),
        - subject orders (strings, e.g. 'order1'), 
        - analysis label (string, name of folder within EMBODY root directory), 
        - other variables: brainproc, maskName, shiftTRs, PENALTY, featSel.

- *step1_machinelearning.m*
    - No necessary edits.

- *batch_embody_step2.m*
    - Define: 
        - path to EMBODY root directory,
        - subject IDs (strings, e.g. '101'),
        - subject orders (strings, e.g. 'order1'), 
        - analysis label (string, name of folder within EMBODY root directory), 
        - other variables: brainproc, maskName, shiftTRs, PENALTY, featSel.

- *step2_machinelearning.m*
    - No necessary edits.
    
**2. Step1 Scripts**
- *0_step1_setup_directories.ipynb*
    - Creates required directory structure within analysis folder.
        - Set the path to EMBODY study root directory (string).
        - Define machine learning analysis label (string).
        - Define directories for compiled stats within each analysis (list of strings).
        
- *1_step1_compile_accuracy.ipynb*
    - Compiles and outputs Step1 classifier accuracy
        - Set path to EMBODY study root directory (string).
        - Define analysis labels (list of strings).
        - Define subject IDs (list of integers).
        - Define subject groups (list of strings).
        
- *2_step1_confusion_matrix.ipynb*
    - Compiles and outputs Step1 confusion matrix
        - Set path to EMBODY study root directory.
        - Define subject IDs (list of integers).
        - Define analysis labels (lists of strings).
        
- *3_step1_compile_individual_decisions.ipynb*
    - Determines accuracy at the individual subject level
        - Set path to EMBODY study root directory (string).
        - Define analysis labels (list of strings).
        - Define subject IDs (list of integers).

- *4_step1_trial_accuracy_ratings.ipynb*
    - Calculates classifier accuracy for each trial where the subject also rates how well they paid attention to the specified task.
        - Set path to EMBODY study root directory (string).
        - Set path to regressor files (string).
        - Set path to EPRIME ratings files (string).
        - Define subject IDs (list of integers).
        - Define subject orders (list of strings, list order must correspond to subject ID list order).


**3. Step2 Scripts**
- *1_step2_3cat_script.ipynb*
    - Decodes meditation into 3 categories: breath, mind-wandering, and self-referential processing.
        - Set path to EMBODY study root directory (string).
        - Set analysis label (string).
        - Define subject IDs (list of integers).

- *2_step2_smoothing.ipynb*
    - Smooths isolated classifications.
        - Set path to EMBODY study root directory (string).
        - Set analysis label (string).
        - Define subject IDs (list of integers).

- *3_step2_event_cleaning.ipynb*
    - Defines events as >= 3 contiguous classifications.
        - Set path to EMBODY study root directory (string).
        - Set analysis label (string).
        - Define subject IDs (list of integers).

**4. Step3 Scripts**
- *1_step3_conditionStats_cleanAll.ipynb*
    - Computes attention metrics during meditation (Breath, Mind-wandering, Self-referential processing)
        - Set path to EMBODY study root directory (string).
        - Set analysis label (string).
        - Define subject IDs (list of integers).
        - Define group membership (meditator or control, list of strings).

- *2_step3_distraction_fluctuation.ipynb*
    - Counts the number of "event switches" from one type of event to another. Calculates the average number of seconds subject was distracted from their breath.
        - Set path to EMBODY study root directory (string).
        - Set analysis label (string).
        - Define subject IDs (list of integers).

- *3_step3_aggregateStats_cleanAll.ipynb*
    - Aggregates statistics across conditions.
        - Set path to EMBODY study root directory (string).
        - Set analysis label (string).
        - Define subject IDs (list of integers).
        - Define group membership (meditator or control, list of strings).

- *4_step3_data_concat.ipynb*
    - Concatenates data files from step3 to enter into master csv file.
        - Set path to EMBODY study root directory (string).
        - Set analysis label (string)

-------------------

### Demo

Within each of the demo scripts you must edit your EMBODY study root directory. For the demo, your **root_dir** will be *./EMBODY_CODE_AND_DEMO/embody_demo/embody_study*

** Note: running the machine learning scripts is optional for the demo. There are sample output files for this step included so you can start out with the iPython Notebook code. If you would like to run the machine learning scripts, sample fMRI data is available [here](https://ucsf.app.box.com/s/l4g4nj7ryy6s48p6dtc0c9qraf04za3y)**

1 (Optional) Run the demo machine learning scripts in MATLAB.
    - Run *./embody_demo/embody_study/scripts/batch_scripts/batch_embody_step1.m*
    - Run *./embody_demo/embody_study/scripts/batch_scripts/batch_embody_step2.m*

2. Run the iPython Notebook code in in the order listed above. The code within the directory *./EMBODY_CODE_AND_DEMO/embody_scripts* is initially set up to run the demo.

-------------------

Additional Information
- Data for participants who consented to the release of their machine learning output is located in the **consented_data** folder. You can access fMRI data that has participants have consented to share [here](https://ucsf.app.box.com/s/l4g4nj7ryy6s48p6dtc0c9qraf04za3y).
- Refer to METHODS & SUPPLEMENTAL INFORMATION. Each Python script describes its
  functionality.
- Disclaimer: Each person will have to edit the scripts to fit their own file
  directory structures & file formatting. Authors are not available for debugging.
