import pandas as pd


def build_features(df: pd.DataFrame):
    """
    Build feature matrix X and target vector y for congestion risk prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing orbital parameters.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    """
    df = df.copy()

    # Validate required columns
    required_cols = {
        "altitude_km",
        "inclination",
        "orbital_band",
        "congestion_risk",
        "eccentricity",
        "mean_motion",
    }

    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    # Feature engineering
    # Orbital density per band (proxy for congestion)
    orbital_density = df.groupby("orbital_band").size()
    df["orbital_density"] = df["orbital_band"].map(orbital_density)

    # Encode orbital band as ordinal category
    orbit_band_mapping = {
        "LEO": 0,
        "MEO": 1,
        "GEO": 2,
    }
    df["orbit_band_code"] = df["orbital_band"].map(orbit_band_mapping)

    # Feature matrix and target
    feature_cols = [
        "altitude_km",
        "inclination",
        "eccentricity",
        "mean_motion",
        "orbital_density",
        "orbit_band_code",
    ]

    X = df[feature_cols]
    y = df["congestion_risk"]

    return X, y
