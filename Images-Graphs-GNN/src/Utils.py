from pathlib import Path
import HyperParameters
import Utils as U

PROJECT_DIR = Path(__file__).parent.parent
CLEAN_DATA_FOLDER = (PROJECT_DIR / 'data_clean').resolve()
RAW_DATA_FOLDER = (PROJECT_DIR / 'data_raw').resolve()
TEST_DATA_FOLDER = (PROJECT_DIR / 'data_test').resolve()
MODEL_FOLDER = (PROJECT_DIR / 'Model').resolve()

training_folders = [(U.RAW_DATA_FOLDER / 'seg_train' / 'seg_train' / class_name.lower()).resolve() 
                        for class_name in HyperParameters.CLASSES]
testing_folders = [(U.RAW_DATA_FOLDER / 'seg_test' / 'seg_test' / class_name.lower()).resolve()
                       for class_name in HyperParameters.CLASSES]
