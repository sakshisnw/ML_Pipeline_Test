from src.pipeline import parse, extract, dataset, clean_data, preprocess, TTS, t_model,e_model

def main():
    path = "algoparams_from_ui.json.json"  
    config = parse(path)
    details = extract(config) 
    
    data = "data/iris.csv"
    df = dataset(data)
    df = clean_data(df)
    features, target = preprocess(df, details["target"])
    
    X_train, X_test, y_train, y_test = TTS(features, target)
    model = t_model(details["sel_algorithm"], X_train, y_train)
    result = e_model(model, X_test, y_test)
    
    print("Model Evaluation Results:", result)
    
    #print("Target Variable:", details["target"])
    #print("Prediction Type:", details["prediction_type"])
    #print("Feature Handling Instructions:", details["feature_handling"])
    #print("Selected Algorithms:", details["sel_algorithm"])
     
    #print("Sample Dataset of iris\n",df.head(5))
    #print("Missing values in iris dataset",df.isnull().sum().sum())
    
if __name__ == "__main__":
    main()
