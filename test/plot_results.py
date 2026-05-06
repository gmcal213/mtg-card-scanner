import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# This script will consume the results .csv and generate a heat map for each test present
def main():
    parser = argparse.ArgumentParser(description="Plot results of run_tests")
    parser.add_argument("--tests_file", help="Test file you want to plot", required=True)
    args = parser.parse_args()

    tests_csv = Path(args.tests_file)
    
    df = pd.read_csv(tests_csv)

    # enumerate each combination of pipeline
    df["pipeline"] = (
        df["edge"].astype(str) + "_" +
        df["thresh"].astype(str) + "_" +
        df["blur"].astype(str)
    )

    # enumarate each combination of scene
    df["condition"] = df["background"].astype(str) + "_" + df["position"].astype(str)

    for deg, sub_df in df.groupby("degredation"):

        # Aggregate CER data and plot
        output_png = tests_csv.with_name(f"{tests_csv.stem}_cer_{deg}.png")

        cer_pivot = sub_df.pivot_table(
            index="pipeline",
            columns="condition",
            values="cer",
            aggfunc="mean"
        )

        title = f"CER by Pipeline and Condition - Degradation: {deg}"

        plt.figure(figsize=(12, 8))
        sns.heatmap(cer_pivot, cmap="Reds", annot=False)
        plt.title(title)

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()

        # Aggregate Confidence data and plot 
        output_png = tests_csv.with_name(f"{tests_csv.stem}_confidence_{deg}.png")

        conf_pivot = sub_df.pivot_table(
            index="pipeline",
            columns="condition",
            values="confidence",
            aggfunc="mean"
        )

        title = f"Confidence by Pipeline and Condition - Degradation: {deg}"

        plt.figure(figsize=(12, 8))
        sns.heatmap(conf_pivot, cmap="Greens", annot=False)
        plt.title(title)

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()

        # Aggregate time data and plot
        output_png = tests_csv.with_name(f"{tests_csv.stem}_time_{deg}.png")

        time_pivot = sub_df.pivot_table(
            index="pipeline",
            columns="condition",
            values="time",
            aggfunc="mean"
        )

        title = f"Time by Pipeline and Condition - Degradation: {deg}"

        plt.figure(figsize=(12, 8))
        sns.heatmap(time_pivot, cmap="Reds", annot=False)
        plt.title(title)

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()