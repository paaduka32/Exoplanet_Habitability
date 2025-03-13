import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

df = pd.read_csv("exoplanet_dataset.csv")

print("Initial dataset info:")
print(df.info())

# Step 1: Drop columns with too many missing values (>50% missing)
missing_threshold = 0.5 * len(df)
df = df.dropna(thresh=missing_threshold, axis=1)

# Step 2: Fill categorical missing values with mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 3: Interpolation for time-related / ordered variables
interp_cols = ["P_PERIOD", "P_TEMP_EQUIL", "P_TEMP_SURF"]
for col in interp_cols:
    if col in df.columns:
        df[col] = df[col].interpolate(method="linear")

# Step 4: Regression-based imputation for related datasets
regression_pairs = [("P_RADIUS", "P_MASS"), ("S_LUMINOSITY", "S_MASS")]
imputer = SimpleImputer(strategy="mean")
for target, predictor in regression_pairs:
    if target in df.columns and predictor in df.columns:
        df[predictor] = imputer.fit_transform(df[[predictor]])

        known_data = df.dropna(subset=[target, predictor])
        missing_rows = df[target].isnull()

        if known_data.empty:
            print(f"⚠️ Skipping {target} (no valid {predictor} values for training).")
            continue  

        valid_missing_rows = missing_rows & df[predictor].notnull()
        if valid_missing_rows.sum() == 0:
            print(f"⚠️ Skipping {target} (no predictor values to use for prediction).")
            continue

        model = LinearRegression()
        model.fit(known_data[[predictor]], known_data[target])

        df.loc[valid_missing_rows, target] = model.predict(df.loc[valid_missing_rows, [predictor]])

# Step 5: Replace zero values with NaN in critical columns
zero_replace_cols = [
    "P_MASS", "P_RADIUS", "P_PERIOD", "P_SEMI_MAJOR_AXIS", "P_ECCENTRICITY",
    "P_INCLINATION", "P_GRAVITY", "P_DENSITY", "P_FLUX", "P_TEMP_EQUIL", "P_TEMP_SURF",
    "S_TEMPERATURE", "S_MASS", "S_RADIUS", "S_LUMINOSITY", "S_SNOW_LINE"
]
for col in zero_replace_cols:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

# Step 6: KNN Imputation for multi-column missing values
knn_imputer = KNNImputer(n_neighbors=5)
df[df.select_dtypes(include=[np.number]).columns] = knn_imputer.fit_transform(df.select_dtypes(include=[np.number]))

# Step 7: Remove duplicate rows if any
df.drop_duplicates(inplace=True)

# Step 8: Consistent units
terran_fix = (df["P_TYPE"] == "Terran") & (~df["P_MASS"].between(0.5, 5))
superterran_fix = (df["P_TYPE"] == "Superterran") & (~df["P_MASS"].between(1.9, 10))
neptunian_fix = (df["P_TYPE"] == "Neptunian") & (~df["P_MASS"].between(5, 10))
jovian_fix = (df["P_TYPE"] == "Jovian") & (df["P_MASS"] < 10)
mass_fix = terran_fix | superterran_fix | neptunian_fix | jovian_fix
df.loc[mass_fix, "P_MASS"] *= 317.8
df.loc[mass_fix, "P_RADIUS"] *= 11.2
df["P_DENSITY"] = (3 * df["P_MASS"]) / (4 * np.pi * (df["P_RADIUS"] ** 3))

print(df.info())
df.to_csv("cleaned_dataset.csv", index=False)
print(df.columns)