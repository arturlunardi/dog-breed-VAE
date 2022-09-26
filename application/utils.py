import os
import shutil

# data configs
_root_data_path = r"../../data"

_train_data_directory = r"train"
_test_data_directory = r"test"

_full_data_directory = r"full_data"

# model configs
image_size = 128
latent_dim = 128
batch_size = 32
n_epochs=10
input_shape = (image_size, image_size, 3)

def join_data():
    if not os.path.exists(os.path.abspath(os.path.join(__file__, _root_data_path, _full_data_directory))):
        os.mkdir(os.path.abspath(os.path.join(__file__, _root_data_path, _full_data_directory)))

    # copy files from train directory
    train_files = 0
    for file_name in os.listdir(os.path.abspath(os.path.join(__file__, _root_data_path, _train_data_directory))):
        source = os.path.abspath(os.path.join(__file__, _root_data_path, _train_data_directory, file_name))
        destination = os.path.abspath(os.path.join(__file__, _root_data_path, _full_data_directory))

        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            train_files += 1
    print(f"Copied {train_files} files")

    # copy files from test directory
    test_files = 0
    for file_name in os.listdir(os.path.abspath(os.path.join(__file__, _root_data_path, _test_data_directory))):
        source = os.path.abspath(os.path.join(__file__, _root_data_path, _test_data_directory, file_name))
        destination = os.path.abspath(os.path.join(__file__, _root_data_path, _full_data_directory))

        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            test_files += 1
    print(f"Copied {test_files} files")

    return None