import json

def parse(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def extract(config):
    target = config["design_state_data"]["target"]["target"]
    prediction_type = config["design_state_data"]["target"]["prediction_type"]
    feature_handling = config["design_state_data"]["feature_handling"]
    
    sel_algorithm = {}
    for k, v in config["design_state_data"]["algorithms"].items():
        if v["is_selected"]:
            sel_algorithm[k]=v

    return {
        "target": target,
        "prediction_type": prediction_type,
        "feature_handling": feature_handling,
        "sel_algorithm": sel_algorithm,
    }