from typing import Tuple, List
import pandas as pd
import numpy as np


def load_feature_tables(
    radiomics_csv: str,
    feat2d_csv: str,
    feat3d_csv: str,
    clinical_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Expect each feature csv has an 'ID' column.
    Returns dataframes indexed by ID.
    """
    r = pd.read_csv(radiomics_csv).set_index("ID")
    f2d = pd.read_csv(feat2d_csv).set_index("ID")
    f3d = pd.read_csv(feat3d_csv).set_index("ID")
    c = pd.read_csv(clinical_csv).set_index("ID")
    return r, f2d, f3d, c


def load_labels(label_csv: str) -> pd.DataFrame:
    """
    label_csv should include columns:
      - ID
      - group  (train/test or train/val/test)
      - label  (int class index)
    """
    return pd.read_csv(label_csv)


def extract_by_ids(df, id_list: List[str]) -> np.ndarray:
    return df.loc[id_list].values


def get_split_ids(label_df: pd.DataFrame, train_key="train", test_key="test"):
    train_ids = label_df[label_df["group"] == train_key]["ID"].tolist()
    test_ids = label_df[label_df["group"] == test_key]["ID"].tolist()
    y_train = label_df[label_df["group"] == train_key]["label"].values
    y_test = label_df[label_df["group"] == test_key]["label"].values
    return train_ids, test_ids, y_train, y_test
