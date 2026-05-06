import csv
import os
import time
import argparse
import itertools
import cv2
import jiwer
import albumentations as A
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from python_scanner.card_scanner import scan_card

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Parse input test csv
def parse_test_fieds(value):
    if not value:  # handles empty fields like blur
        return []
    return value.split(",") 

# Filters input cards csv
def row_matches(row, filters):
    for col, allowed_values in filters.items():
        if allowed_values is None:
            continue
        if row[col] not in allowed_values:
            return False
    return True

# build the experiment list for test
def build_experiments(card_set, preprocess_types, degredation_types):
    
    experiments = []

    preprocess_steps = ["blur", "edge", "thresh"]

    preprocess_values = [
        preprocess_types[step] if preprocess_types[step] else [None]
        for step in preprocess_steps
    ]

    # enumarate each combination of card, preprocessing pipeline, and degradation
    for card in card_set:
        for deg in degredation_types:
            for preprocess_pipeline in itertools.product(*preprocess_values):

                experiment = {
                    "card": card,
                    "degredation": deg,
                    "preprocess": {
                        "blur": preprocess_pipeline[0],
                        "edge": preprocess_pipeline[1],
                        "thresh": preprocess_pipeline[2]
                    }
                }

                experiments.append(experiment)

    return experiments

# apply image degradation
def image_degrade(input_image, option):

    # apply specified degredation
    if(option == "blur"):
        transform = A.GaussianBlur(blur_limit=(3,5), p=1.0)
    elif(option == "noise"):
        transform = A.GaussNoise(std_range=(0.02, 0.08), p=1.0)
    elif(option == "lighting"):
        transform = A.RandomBrightnessContrast(0.2, 0.2, p=1.0)
    elif(option == "color"):
        transform = A.HueSaturationValue(5, 5, 5, p=1.0)
    elif(option == "compression"):
        transform = A.ImageCompression(quality_range=(70, 100), p=1.0)
    elif(option == "occlusion"):
        transform = A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 64),hole_width_range=(16, 64), p=1.0)
    elif(option == "distortion"):
        transform = A.GridDistortion(distort_limit=0.05, p=1.0)
    else:
        return input_image # none or wrong degradation specified
    
    output_image = transform(image=input_image)['image']

    return output_image

# runs a single test
def run_single_test(exp):
        
        cv2.setNumThreads(0)
        # build file path
        image_file = os.path.join(BASE_DIR, "images", exp["card"]["file"])

        # Read in Image
        image = cv2.imread(image_file)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_file}")

        # Apply selected degredation
        image = image_degrade(image, exp["degredation"])

        # Perform time keeping
        start = time.perf_counter()
        # Run card scanner
        title, confidence = scan_card(image, exp["preprocess"]["blur"], exp["preprocess"]["edge"], exp["preprocess"]["thresh"])
        # End timer
        end = time.perf_counter()

        # Calculate Character Error Rate (CER)
        cer = jiwer.cer(exp["card"]["title"], title)

        return {
            "experiment": exp,
            "cer": cer,
            "confidence": confidence,
            "time": end - start
        }

# runs the full test suite
def run_tests(experiments):

    results = []
    total = len(experiments)
    
    workers = min(os.cpu_count() or 4, 6) # keep OCR stable
    
    # spawn worker threads to complete tests
    with ProcessPoolExecutor(max_workers=workers) as executor:

        # list of experiment calls
        future_to_exp = {
            executor.submit(run_single_test, exp): exp
            for exp in experiments
        }

        # for each test that is completed
        for i, future in enumerate(as_completed(future_to_exp), 1):
            exp = future_to_exp[future]

            # append results
            print(f"\rCompleted {i}/{total}", end="", flush=True)
            try:
                result = future.result(timeout=10)
                results.append(result)
            except Exception as e:
                print("\n--- Test Failure ---")
                print(f"Experiment: {exp}")
                print(f"{repr(e)}")

    return results


# builds unqiue output path for file
def get_unique_output_path(base_path):
    base, ext = os.path.splitext(base_path)

    # get and add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_path = f"{base}_{timestamp}{ext}"

    return new_path

# build final csv
def build_csv(results, output_file):
    fieldnames = [
        "id",
        "card_title",
        "background",
        "position",
        "degredation",
        "blur",
        "edge",
        "thresh",
        "cer",
        "confidence",
        "time"
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        i = 0

        # for each experiment result
        for r in results:
            i = i + 1

            # write the test parameters and results
            exp = r["experiment"]
            card = exp["card"]
            preprocess = exp["preprocess"]

            row = {
                "id": i,
                "card_title": card.get("title"),
                "background": card.get("background"),
                "position": card.get("position"),
                "degredation": exp["degredation"],
                "blur": preprocess["blur"],
                "edge": preprocess["edge"],
                "thresh": preprocess["thresh"],
                "cer": r["cer"],
                "confidence": r["confidence"],
                "time": r["time"]
            }

            writer.writerow(row)

    return

# This script runs the test pipeline for mtg card scanner
# It organizes the group of data 
def main():
    parser = argparse.ArgumentParser(description="Test the MTG Card Scanner App")
    parser.add_argument("--tests_file", required=True)
    parser.add_argument("--cards_file", required=True)
    args = parser.parse_args()

    tests_csv = args.tests_file
    cards_csv = args.cards_file

    output_dir = os.path.join(BASE_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    results_csv = os.path.join(output_dir, "results.csv")

    # Read tests file
    with open(tests_csv, newline="") as f:
        reader = csv.DictReader(f)       
        test_set = []
        for row in reader:
            parsed = {
                "id": int(row["id"]),
                "background": parse_test_fieds(row["background"]),
                "position": parse_test_fieds(row["position"]),
                "blur": parse_test_fieds(row["blur"]),
                "edge": parse_test_fieds(row["edge"]),
                "thresh": parse_test_fieds(row["thresh"]),
                "degredation": parse_test_fieds(row["degredation"]),
            } # parse test cases
            test_set.append(parsed) 

    # read in cards set
    with open(cards_csv, newline="") as f:
        reader = csv.DictReader(f)
        all_cards = list(reader)

    total = len(test_set)

    # Iterate through each test in the tests csv
    for i, test in enumerate(test_set, 1):

        print(f"Running Test Set: {i}/{total}")

        # Extract all appropriate cards from cards csv
        image_filters = {
            "background": test["background"],
            "position": test["position"]
        }

        # filter appropriate cards for test set
        card_set = [
            row for row in all_cards
            if row_matches(row, image_filters)
        ]

        # Enumerate preprocessing pipelines
        preprocess_types = {
            "blur": test["blur"],
            "edge": test["edge"],
            "thresh": test["thresh"]
        }

        # retrieve the degradation type
        degredation_types = test["degredation"]

        # build experiments for test
        experiments = build_experiments(card_set, preprocess_types, degredation_types)

        # Run the test
        results = run_tests(experiments) 

        # get unique csv name
        results_csv_file = get_unique_output_path(results_csv)   

        print(f"\nWriting results to: {results_csv_file}")

        # build output csv file
        build_csv(results, results_csv_file)

if __name__ == "__main__":
    main()