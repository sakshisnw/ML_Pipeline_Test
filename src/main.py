from src.pipeline import parse, extract

def main():
    path = "algoparams_from_ui.json.json"  
    
    config = parse(path)
    
    details = extract(config)  
    
    print("Target Variable:", details["target"])
    print("Prediction Type:", details["prediction_type"])
    print("Feature Handling Instructions:", details["feature_handling"])
    print("Selected Algorithms:", details["sel_algorithm"]) 

    
if __name__ == "__main__":
    main()
