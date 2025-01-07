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
            sel_algorithm[k] = v

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

def feature_reduction(features, method, target=None):
    if method == "PCA":
        pca = PCA(n_components=min(features.shape[1], 5))
        features = pca.fit_transform(features)
    elif method == "CorrWithTarget":
        if target is not None:
            selector = SelectKBest(k=min(features.shape[1], 5))
            features = selector.fit_transform(features, target)
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