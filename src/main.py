import json
import pandas as pd
from pipeline import parse, extract, dataset, clean_data, preprocess, TTS, feature_reduction, t_model, e_model

def main():
    # Load 
    config_path = "algoparams_from_ui.json.json"  
    data_path = "data/iris.csv" 

    # Parse the JSON config
    config = parse(config_path)
    extracted = extract(config)
    
    # Extract parameters 
    target = extracted["target"]
    prediction_type = extracted["prediction_type"]
    feature_handling = extracted["feature_handling"]
    sel_algorithm = extracted["sel_algorithm"]

    # Load and clean dataset
    df = dataset(data_path)
    print("Dataset Loaded:\n", df.head())
    print("\nDataset shape:\n", df.shape)
    
    df = clean_data(df)

    # Preprocess the dataset
    features, target_data = preprocess(df, target)
    print(f"Features shape: {features.shape}")
    
    # Apply feature reduction 
    if "reduction_method" in feature_handling:
        features = feature_reduction(features, feature_handling["reduction_method"], target_data)

    # Split data
    X_train, X_test, y_train, y_test = TTS(features, target_data)

    # Train models 
    trained_models = t_model(sel_algorithm, X_train, y_train)

    # Evaluate models 
    results = e_model(trained_models, X_test, y_test)

    # Print the results
    print("\nModel Evaluation Results")
    for model_name, predictions in results.items():
        print(f"{model_name}: {predictions}")

if __name__ == "__main__":
    main()
