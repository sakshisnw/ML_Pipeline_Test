import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


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
    
def dataset(data):
    return pd.read_csv(data)

def clean_data(df):
    if df.isnull().sum().sum() > 0:
        df = df.dropna()    
    return df

def preprocess(df, target):
    features = df.drop(columns=[target])
    t_val = df[target]
    
    if t_val.dtypes == 'object':
        encoder = LabelEncoder()
        t_val = encoder.fit_transform(t_val)
        
    cat_cols = features.select_dtypes(include=['object']).columns
    num_cols = features.select_dtypes(exclude=['object']).columns
    
    if len(cat_cols) > 0:
        features[cat_cols] = features[cat_cols].apply(LabelEncoder().fit_transform)
    
    scaler = StandardScaler()
    features[num_cols] = scaler.fit_transform(features[num_cols])
     
    return features, t_val


def TTS(features, target):
    return train_test_split(features, target, random_state=40, test_size=0.2)

def t_model(m_config, X_train, y_train):
    trained = {}
    for m_name, params in m_config.items():
        if m_name == "RandomForestRegressor":
            valid_params = {k: v for k, v in params.items() if k in RandomForestRegressor().get_params()}
            model = RandomForestRegressor(**valid_params)
        elif m_name == "LinearRegression":
            model = LinearRegression()
        else:
            continue
        model.fit(X_train, y_train)
        t_model[m_name] = model
    return trained

def e_model(models, X_test, y_test):
    result = {}
    for m_name, model in models.items():
        pred = model.predict(X_test)
        result[m_name] = pred
    return result

def e_model():
    return