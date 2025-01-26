import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

def parse(path):
    with open(path, 'r') as file:
        config = json.load(file)
    
    # Define expected keys
    required_keys = ['target', 'prediction_type', 'feature_handling', 'algorithms']
    
    missing_keys = [key for key in required_keys if key not in config["design_state_data"]]
    
    if missing_keys:
        print(f"Warning: Missing required keys in JSON: {missing_keys}")
        for key in missing_keys:
            if key == 'target':
                config["design_state_data"][key] = {}  
            elif key == 'prediction_type':
                config["design_state_data"][key] = "Regression" 
            elif key == 'feature_handling':
                config["design_state_data"][key] = {}  
            elif key == 'models':
                config["design_state_data"][key] = []  
    
    return config


def extract(config):
    try:
        target = config["design_state_data"]["target"]["target"]
        prediction_type = config["design_state_data"]["target"]["prediction_type"]
        feature_handling = config["design_state_data"]["feature_handling"]

        sel_algorithm = {}
        for k, v in config["design_state_data"]["algorithms"].items():
            if v.get("is_selected", False):
                sel_algorithm[k] = v

    except KeyError as e:
        print(f"Warning: Missing key {e} in the configuration, skipping it.")
        target, prediction_type, feature_handling, sel_algorithm = None, "Classification", {}, {}

    return {
        "target": target,
        "prediction_type": prediction_type,
        "feature_handling": feature_handling,
        "sel_algorithm": sel_algorithm,
    }


def dataset(data):
    return pd.read_csv(data)


def clean_data(df):
    # Check and handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()  # Dropping rows with missing values
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


def feature_reduction(features, method, target=None):
    if method == "PCA":
        pca = PCA(n_components=min(features.shape[1], 5))
        features = pca.fit_transform(features)
    elif method == "CorrWithTarget":
        if target is not None:
            selector = SelectKBest(k=min(features.shape[1], 5))
            features = selector.fit_transform(features, target)
    elif method == "Tree-based":
        if target is not None:
            rf_selector = RandomForestRegressor(n_estimators=100)
            rf_selector.fit(features, target)
            importances = rf_selector.feature_importances_
            indices = importances.argsort()[-5:][::-1]
            features = features.iloc[:, indices]
    return features


def t_model(m_config, X_train, y_train):
    trained_models = {}
    for m_name, params in m_config.items():
        if m_name == "RandomForestRegressor":
            model = RandomForestRegressor(random_state=42)
        elif m_name == "LinearRegression":
            model = LinearRegression()
        else:
            continue

        grid_params = params.get("grid_search", None)
        if grid_params:
            grid_search = GridSearchCV(estimator=model, param_grid=grid_params, cv=3)
            grid_search.fit(X_train, y_train)
            trained_models[m_name] = grid_search.best_estimator_
        else:
            model.fit(X_train, y_train)
            trained_models[m_name] = model

    return trained_models


def e_model(models, X_test, y_test):
    result = {}
    for m_name, model in models.items():
        pred = model.predict(X_test)
        result[m_name] = pred
    return result


def create_pipeline(nf, cf):
    num_tran = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_tran = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_tran, nf),
            ('cat', cat_tran, cf)
        ]
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))
    ])
    return pipeline
