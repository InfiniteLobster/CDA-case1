#--------------------Libraries--------------------#
#hadnling of command line arguments and paths
import argparse
from email import parser
from pathlib import Path
#handling of the data and models
import pandas as pd
import joblib
#--------------------Functions--------------------#
#function to get absolute paths to files (if already not given)
def resolve_path(base_path: Path, configured_path: str | None) -> Path | None:
    #changing the path to a Path object for easier handling
    path_file = Path(configured_path)
    #making the path absolute (if it is not already) by resolving it from the base path (project root or data folder, depending on how the path is given in the command line arguments)
    if not path_file.is_absolute():
        path_absolute = base_path / path_file
    else:
        path_absolute = path_file
    #resolve the path (it is needed due to the Pathlib's behavior of not resolving the path when it is created with the / operator, even if the base path is absolute)
    path_absolute = path_absolute.resolve()        
    #return output
    return path_absolute
#function to handle the input from command line arguments
def parse_args() -> argparse.Namespace:
    #creating the parser for command line arguments
    parser = argparse.ArgumentParser(
        description="Load a trained model and generate predictions for a CSV dataset."
    )
    #adding argument for the data path (to make predictions on)
    parser.add_argument(
        "--data-path",
		dest="data_path",
		required=True,
		help="Path to the CSV file to score.",
	)
    #adding argument for the model path (to make predictions with)
    parser.add_argument(
		"--model-path",
		dest="model_path",
		required=True,
		help="Path to an outputs run directory or directly to best_model.joblib.",
	)
    #adding argument for the output path (to save the predictions in different than assumed one(predictions folder in the project root))
    parser.add_argument(
		"--output-path",
		dest="output_path",
		help="Optional path for the predictions CSV. Relative paths are resolved from the input data folder.",
	)
    #setting the parsing to Namespace for easier handling of the arguments downstream (instead of the default list of strings)
    parser_out = parser.parse_args()
    #returning output
    return parser_out

#this is function which takes care of predictions by the model on the data
def make_predictions(data_path, model_path):
	#importing code form modules (seprate files for better orgnization of logic)
	from data import load_data_predi
	#loading the data to make predictions from
	data = load_data_predi(data_path)
	#loading the model to make predictins with
	model = joblib.load(model_path)
	#generating predictions
	predictions = model.predict(data)
	#returning output
	return predictions
#--------------------Code--------------------#
def main() -> None:
    #getting arguments from command line
    args = parse_args()
    #getting project root path to resolve relative paths from configs and command line arguments
    project_root = Path(__file__).resolve().parents[2]
    #getting the absolute paths to the files for prediction
    data_path = resolve_path(project_root, args.data_path)
    model_path = resolve_path(project_root, args.model_path)
    #checking if given paths actually exist
    if not data_path.exists():
        raise FileNotFoundError(f"Data for predictions was not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model for predictions was not found: {model_path}")
    #getting the predictions
    predictions = make_predictions(data_path, model_path)
    #getting the output path (if not given, make the generic)
    if args.output_path is None:
        predictions_dir = project_root / "predictions"
        output_path = predictions_dir / f"{model_path.stem}_predictions.csv"
    else:
        output_path = resolve_path(data_path.parent, args.output_path)
	#making sure the output directory exists (if not, it will be created)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    #saving the predictions to a csv file
    output = pd.DataFrame({"y_pred": predictions})
    output.to_csv(output_path, index=False, header=False)
    #printing some info to user
    print(f"Loaded model: {model_path}")
    print(f"Scored data: {data_path}")
    print(f"Saved predictions: {output_path}")
    print(f"Rows predicted: {len(output)}")


if __name__ == "__main__":
	main()
