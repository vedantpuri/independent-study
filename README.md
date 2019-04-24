# Independent Study

This is a repository containing the work for my Independent Study under professor Andrew McCallum
in IESL.

## Task

At a high level I was involved in the task of Shallow Semantic Parsing for Scientific Procedural Texts. Within this I was responsible for implementing and running a supervised learning model using PyTorch. Apart from a function or two, majority of the code has been written from scratch.

## Data

### Formatting

## Requirements
- Python >= **3**
- PyTorch >= **1.0.1**
- Scikit-learn >= **0.18.1**


## Configuration
The configuration file must be defined as a single json dict. The following are the fields for the configuration file along with their requirement statuses.

- ####  TRAINING_FILE_PATH (*String*)
  Path to the file where the training data is stored. <br>[**Required**]
- ####  DEV_FILE_PATH (*String*)
  Path to the file where the development data is stored. <br>[**Required**]
- ####  TEST_FILE_PATH (*String*)
  Path to the file where the testing data is stored. <br>[**Required**]
- ####  MODEL_FILE (*String*)
  Path where the model is supposed to be dumped/ loaded from depending on the switches <br>[**Required**]
- ####  TRAIN_MODE (*Boolean*)
  A switch specifying whether the model should train on data or not <br>[**Required**]
- ####  NUM_EPOCHS (*Integer >= 10*)
  Number of passes to make over training data while training <br>[**Required** only if TRAIN_MODE is *true* ]
- ####  DROPOUT_P (*Float in range [0,1]*)
  Dropout probability while training <br>[**Required** only if TRAIN_MODE is *true* ]
- ####  LEARNING_RATE (*Float in range (0,1)*)
  Learning rate for training the model (step size) <br>[**Required** only if TRAIN_MODE is *true* ]
- ####  EMBED_SIZE (*Integer >= 20*)
  Embedding dimension for Predicates and Arguments (same size used for both) <br>[**Required** only if TRAIN_MODE is *true* ]
- ####  LINEARITY_SIZE (*Integer < 2 X EMBED_SIZE*)
  Linear mapping dimension for the concatenated predicate argument embedding <br>[**Required** only if TRAIN_MODE is *true* ]
- ####  BATCH_SIZE (*Integer >= 5*)
  Number of passes to make over training data while training <br>[**Required** only if TRAIN_MODE is *true* ]
- ####  CHECK_EVERY (*Integer >= 3*)
  Number specifying the frequency of running model on dev set while training. Model is run on dev set every CHECK_EVERY   iterations <br>[**Required** only if TRAIN_MODE is *true* ]
- ####  TEST_MODE
  A switch specifying whether the model should predict on data or not <br>[**Required**]
- ####  DUMP_TEST_PREDS
  Switch to specify whether write the predictions made on test set to a file (predictions.txt) <br>
[**Required** only if TEST_MODE is *true* ]
- ####  RUN_BASELINES
  Switch to specify whether to run baseline models or not <br>
[**Required** only if TEST_MODE is *true* ]

**Notes:** 
1. One out of TRAIN_MODE or TEST_MODE **must** be switched on in order for the code to do something. If this violated an error will be thrown.
2. If the paths are relative, they **must** be so with respect to driver.py's directory.

For an exact example for formatting this config as a json file have a look at src/config.json


## Usage
Once the configuration has been properly defined, (for ease) place it in the same directory as driver.py. 
Finally, to run the program, simply navigate to the location of "driver.py'' and run the following command:
```bash
python driver.py config.json
```
