import os
import json

class ConfigManager:
    def __init__(self, config_file):
        self.file_path = config_file

    def parse_config(self):
        configuration = json.load(open(self.file_path))

        # Populate variables from config
        self.train_file_path = configuration["TRAINING_FILE_PATH"]
        assert(os.path.exists(self.train_file_path))

        self.dev_file_path = configuration["DEV_FILE_PATH"]
        assert(os.path.exists(self.dev_file_path))

        self.test_file_path = configuration["TEST_FILE_PATH"]
        assert(os.path.exists(self.test_file_path))

        # A file for dumping/loading the model to/from depending on switches
        self.model_file = configuration["MODEL_FILE"]

        self.train_switch = configuration["TRAIN_MODE"]
        self.predict_switch = configuration["TEST_MODE"]
        # At least one of them must be on
        assert(self.train_switch or self.predict_switch)

        # If in only prediction mode, a model file must exist to load from
        if not self.train_switch and self.predict_switch:
            assert(os.path.exists(self.model_file))

        if self.train_switch:
            self.epochs = configuration["NUM_EPOCHS"]
            assert(self.epochs >= 10)

            self.drop_p = configuration["DROPOUT_P"]
            assert(self.drop_p >= 0 and self.drop_p <= 1)

            self.learn_rate = configuration["LEARNING_RATE"]
            assert(self.learn_rate > 0 and self.learn_rate < 1)

            self.embed_size = configuration["EMBED_SIZE"]
            assert(self.embed_size >= 20)

            self.linearity_size = configuration["LINEARITY_SIZE"]
            assert(self.linearity_size < self.embed_size * 2)

            self.batch_size = configuration["BATCH_SIZE"]
            assert(self.batch_size >= 5)

            self.check_every = configuration["CHECK_EVERY"]
            assert(self.check_every >= 3)

        if self.predict_switch:
            self.dump_test_preds = configuration["DUMP_TEST_PREDS"]
            self.run_baselines = configuration["RUN_BASELINES"]
