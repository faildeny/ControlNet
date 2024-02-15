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
        "--keyword_1",
        type=str,
        nargs="?",
        default=None,
        help="Keyword to search for in the first dataset",
    )
    parser.add_argument(
        "--keyword_2",
        type=str,
        nargs="?",
        default=None,
        help="Keyword to search for in the second dataset",
    )
    
    args = parser.parse_args()

    return args

def create_temporary_directory(directory, n_samples, lower_bound=False, only_unique_ids=True, keyword=None):
    # get dir 
    # generate random hash  
    temp_dir = os.path.join("tmp", directory + str(random.getrandbits(128)))
    os.makedirs(temp_dir, exist_ok=False)
    files_list = os.listdir(directory)
    # exclude files with "mask" in name
    files_list = [file for file in files_list if "mask" not in file]
    # Only keep files with keyword in name
    if keyword:
        files_list = [file for file in files_list if keyword in file]
    if len(files_list) < n_samples:
        print("Number of files with keyword", keyword, "in", directory, ":", len(files_list))
        print("Not enough samples. Duplicates will be used as an alternative.")
        # Add duplicates to the list
        files_list = files_list * (2*n_samples // len(files_list) + 1)
        print("Number of files after adding duplicates:", len(files_list))

    # Only keep full file names with unique id in name
    # if only_unique_ids:
    #     files_unique = []
    #     unique_ids = set()
    #     for file in files_list:
    #         # id = file.split("_")[0]
    #         id = file
    #         if id not in unique_ids:
    #             unique_ids.add(id)
    #             files_unique.append(file)
    #     files_list = files_unique

    # Get n random files
    print("Getting", n_samples, "random samples from", len(files_list), "samples")
    random_files_list = random.sample(files_list, n_samples)

    for file in random_files_list:
        shutil.copyfile(
            os.path.join(directory, file),
            os.path.join(temp_dir, file),
        )
        files_list.remove(file)

    if lower_bound:
        # copy the same images to another directory
        temp_dir2 = os.path.join("tmp", directory + str(random.getrandbits(128)))
        os.makedirs(temp_dir2, exist_ok=False)
        random_files_list = random.sample(files_list, n_samples)

        for file in random_files_list:
            shutil.copyfile(
                os.path.join(directory, file),
                os.path.join(temp_dir2, file),
            )
        return temp_dir, temp_dir2
    

    return temp_dir


if __name__ == "__main__":
    args = parse_args()

    directory_1 = args.dataset_path_1
    directory_2 = args.dataset_path_2
    keyword_1 = args.keyword_1
    keyword_2 = args.keyword_2
    # directory_1 = "training/stacked_EDES_fold_0_prev_0_01/target"
    # directory_2 = "training/stacked_EDES_fold_0_prev_0_01/target"
    n_samples = 2000

    if directory_1 == directory_2:
        lower_bound = True
        print("Calculating lower bound of FID between unique ids in the same dataset")
    else:
        lower_bound = False

    if lower_bound:
        temp_dir1, temp_dir2 = create_temporary_directory(directory_1, n_samples, lower_bound=True, keyword=keyword_1)
    else:
        temp_dir1 = create_temporary_directory(directory_1, n_samples, keyword=keyword_1)
        temp_dir2 = create_temporary_directory(directory_2, n_samples, keyword=keyword_2)

    print("Calculated between", temp_dir1, "and", temp_dir2, "with", n_samples, "samples")

    # execute command to calculate FID
    command = f"python -m pytorch_fid {temp_dir1} {temp_dir2}"
    training_process = subprocess.run(command, shell=True)

    # remove temporary directories
    shutil.rmtree(temp_dir1)
    shutil.rmtree(temp_dir2)
