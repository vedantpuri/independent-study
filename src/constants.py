PREDICATE_KEY = "row"
ARGUMENT_KEY = "col"
ROLE_KEY = "col_roles"

TEMP_PREDICTION_FILE = "predictions.txt"

# Print Messages
DIVIDER = "##################################################\n"
CONFIG_PARSE_BEGIN = "Parsing configuration file ..."
CONFIG_PARSE_SUCC = "Configuration parsed successfully.\n"

ENCODING_BEGIN = "Encoding Data (Forming Mappings) ..."
ENCODING_SUCC = "Data successfully encoded.\n"

TRAIN_INV = "Entered Training Mode: Initializing Training tools ..."
TRAIN_INV_SUCC = "Successfully initialized tools.\n"

TRAIN_BEGIN = "Training model ..."
TRAIN_SUCC = "Training complete.\n"

GRAPH_PLOT_MSG = "Training Graphs saved in /../Figures/"
MODEL_DUMP_MSG = "Best Parameters etc. saved to: "

TEST_INV = "Test Mode activated\n"
TEST_PREDS_BEGIN = "Making predictions on test data ..."
PREDS_END = "Finished Making predictions."

WRITE_PREDS_BEGIN = "Writing predictions"
WRITE_PREDS_SUCC = "Predictions successfully written to: "
WRITE_PREDS_SUCC += TEMP_PREDICTION_FILE + "\n"

TEST_REPORT_MSG = "Classification Report for Test Data: \n"

TRAIN_PREDS_BEGIN = "Making predictions on training data ..."
DEV_PREDS_BEGIN = "Making predictions on dev data ..."

TRAIN_METRICS = "Metrics obtained upon running on training data: "
DEV_METRICS = "Metrics obtained upon running on dev data: "

BASE_BEGIN = "Running Baseline Experiments ...\n"
BASE_END = "Baseline Experiments Complete\n"

RAND_PRED = "Making Random Predictions on test data ..."
RAND_METRIC_MSG = "Metrics obtained for Random Prediction model:"

MAJ_PRED = "Making Majority Label Predictions on test data ..."
MAJ_METRIC_MSG = "Metrics obtained for Majority Label Prediction model:"

PR = "Precision: \t"
RE = "Recall: \t"
F1 = "F1: \t\t"

TERMINATED = "\nProgram Ran to completion successfully!"
