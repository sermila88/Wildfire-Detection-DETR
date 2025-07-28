import subprocess
import glob
import argparse

def main(dataset, model_directory, project):
    source = f"{dataset}"
    weights = glob.glob(f"{model_directory}/**/*best.pt", recursive=True)

    print(f"Number of weight files found: {len(weights)}")

    for weight in weights:
        # Extract only the relevant part of the directory name
        model_name = weight.split('/')[-3]
        cmd = f"yolo predict model={weight} iou=0.01 conf=0.01 source={source} save=False save_txt save_conf project=models/{project}/test_preds name={model_name}"
        print(f"* Command:\n{cmd}")
        subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO predictions on validation set for multiple models.")
    parser.add_argument('--data_directory', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--model_directory', type=str, required=True, help='Path to the directory containing model weights.')
    parser.add_argument('--project', type=str, required=True, help='Path to .')


    args = parser.parse_args()
    main(args.data_directory, args.model_directory, args.project)