# Independent Study

This is a repository containing the work for my Independent Study under professor Andrew McCallum
in IESL.

## Task

At a high level I was involved in the task of Shallow Semantic Parsing for Scientific Procedural Texts. Within this I was responsible for implementing and running a supervised learning model using PyTorch. Apart from a function or two, majority of the code has been written from scratch.

## Data
TO DO
### Formatting
TO DO

## Requirements
- Python >= **3**
- PyTorch >= **1.0.1**
- Scikit-learn >= **0.18.1**


## Configuration
The configuration file must be defined as a single json dict. The following are the fields for the configuration file along with their requirement statuses.

| Field        | Description |Type           | Restriction  | Required
| :------------- | :-------------:| :-------------:| :-----:|  :-----:|
| TRAINING_FILE_PATH | Path to the file where the training data is stored | String | File Must Exist | Always|
| DEV_FILE_PATH  | Path to the file where the development data is stored | String |   File Must Exist | Always|
| TEST_FILE_PATH | Path to the file where the testing data is stored | String  |    File Must Exist | Always|
| MODEL_FILE | Path where the model is supposed to be dumped/loaded from depending on the switches | String  |  If in test mode File Must Exist | Always|
| TRAIN_MODE | A switch specifying whether the model should train on data or not| Boolean  |    None | Always|
| NUM_EPOCHS | Number of passes to make over training data while training | Integer  |  >= 10 | Only if in train mode|
| DROPOUT_P | Dropout probability while training | Float  |  In range [0,1] | Only if in train mode|
| LEARNING_RATE | Learning rate for training the model (step size) | Float  |    In range (0,1) | Only if in train mode|
| EMBED_SIZE | Embedding dimension for Predicates and Arguments (same size used for both) | Integer |  >= 20 | Only if in train mode|
| LINEARITY_SIZE | Linear mapping dimension for the concatenated predicate argument embedding| Integer  |  < 2 * EMBED_SIZE | Only if in train mode|
| BATCH_SIZE | Number of passes to make over training data while training| Integer  |  >= 5 | Only if in train mode|
| CHECK_EVERY | Number specifying the frequency of running model on dev set while training. Model is run on dev set every CHECK_EVERY   iterations| Integer  |  >= 3 | Only if in train mode|
| TEST_MODE | A switch specifying whether the model should predict on data or not| Boolean  |    None | Always|
| DUMP_TEST_PREDS | Switch to specify whether write the predictions made on test set to a file (predictions.txt)| Boolean  |    None | Only if in test mode|
| RUN_BASELINES | Switch to specify whether to run baseline models or not| Boolean  |  None | Only if in test mode|



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
