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

        self.epochs = configuration["NUM_EPOCHS"]
        assert(self.epochs > 10)

        self.drop_p = configuration["DROPOUT_P"]
        assert(self.drop_p > 0 and self.drop_p < 1)

        self.learn_rate = configuration["LEARNING_RATE"]
        assert(self.learn_rate > 0 and self.learn_rate < 1)

        self.embed_size = configuration["EMBED_SIZE"]
        assert(self.embed_size > 20)

        self.linearity_size = configuration["LINEARITY_SIZE"]
        assert(self.linearity_size < self.embed_size * 2)

        self.batch_size = configuration["BATCH_SIZE"]
        assert(self.batch_size > 5)

        self.check_every = configuration["CHECK_EVERY"]
        assert(self.check_every > 3)

        self.model_dump_file = configuration["MODEL_DUMP_FILE"]
        self.test_pred_file_destroy = configuration["TEST_PRED_FILE_DESTROY"]
        self.run_baselines = configuration["RUN_BASELINES"]
