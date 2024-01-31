import argparse
import os
import random
import shutil
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculates the Frechet Inception Distance between two distributions using RadImageNet model."
    )
    parser.add_argument(
        "dataset_path_1",
        type=str,
        help="Path to images from first dataset",
    )
    parser.add_argument(
        "dataset_path_2",
        type=str,
        help="Path to images from second dataset",
    )
    parser.add_argument(
        "--lower_bound",
        action="store_true",
        help="Calculate lower bound of FID using the 50/50 split of images from dataset_path_1",
    )
    
    args = parser.parse_args()

    return args

def create_temporary_directory(directory, n_samples):
    # get dir 
    # generate random hash  
    temp_dir = os.path.join("tmp", directory + str(random.getrandbits(128)))
    os.makedirs(temp_dir, exist_ok=False)
    files_list = os.listdir(directory)
    # exclude files with "mask" in name
    files_list = [file for file in files_list if "mask" not in file]
    
    # Get n random files
    files_list = random.sample(files_list, n_samples)
    for file in files_list:
        shutil.copyfile(
            os.path.join(directory, file),
            os.path.join(temp_dir, file),
        )

    return temp_dir


if __name__ == "__main__":
    args = parse_args()

    directory_1 = args.dataset_path_1
    directory_2 = args.dataset_path_2
    lower_bound = args.lower_bound
    n_samples = 5000

    temp_dir1 = create_temporary_directory(directory_1, n_samples)
    temp_dir2 = create_temporary_directory(directory_2, n_samples)

    # execute command to calculate FID
    command = f"python -m pytorch_fid {temp_dir1} {temp_dir2}"
    training_process = subprocess.run(command, shell=True)

    # remove temporary directories
    shutil.rmtree(temp_dir1)
    shutil.rmtree(temp_dir2)
