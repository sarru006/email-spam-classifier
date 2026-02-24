"""
utils/data_loader.py
--------------------
Load and combine the Enron Spam Dataset and the SMS Spam Collection,
deduplicate, and return a clean pandas DataFrame.

Actual file layout (CSV-based):
  data/
    enron_raw/
      enron_spam_data.csv   cols: Subject, Message, Spam/Ham, Date
    sms_raw/
      spam.csv              cols: v1 (label), v2 (text)
"""

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR   = Path(__file__).parent
ENRON_CSV  = DATA_DIR / "enron_spam_data.csv"
SMS_CSV    = DATA_DIR / "sms_data" / "SMSSpamCollection"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_enron() -> pd.DataFrame:
    """
    Read enron_spam_data.csv.
    Relevant columns: 'Subject', 'Message', 'Spam/Ham'
    """
    if not ENRON_CSV.exists():
        logger.warning(f"Enron CSV not found at {ENRON_CSV}")
        return pd.DataFrame(columns=["label", "text", "source"])

    df = pd.read_csv(ENRON_CSV, encoding="utf-8", low_memory=False)
    # Combine Subject + Message as the full email text
    df["text"] = (
        df["Subject"].fillna("") + " " + df["Message"].fillna("")
    ).str.strip()
    df["label"]  = df["Spam/Ham"].str.lower().str.strip()
    df["source"] = "enron"
    df = df[["label", "text", "source"]].dropna(subset=["text"])
    logger.info(f"Enron: {len(df)} emails  {df['label'].value_counts().to_dict()}")
    return df


def _load_sms() -> pd.DataFrame:
    """
    Read spam.csv.
    Columns: v1 = label (ham/spam), v2 = message text
    """
    if not SMS_CSV.exists():
        logger.warning(f"SMS CSV not found at {SMS_CSV}")
        return pd.DataFrame(columns=["label", "text", "source"])

    df = pd.read_csv(SMS_CSV, sep="\t", header=None, names=["label", "text"], encoding="latin-1", low_memory=False)
    df["label"]  = df["label"].str.lower().str.strip()
    df["source"] = "sms"
    df = df[["label", "text", "source"]].dropna(subset=["text"])
    logger.info(f"SMS: {len(df)} messages  {df['label'].value_counts().to_dict()}")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    deduplicate: bool = True,
    min_text_len: int = 10,
) -> pd.DataFrame:
    """
    Load, combine, and clean both datasets.

    Returns
    -------
    pd.DataFrame with columns: label (str), text (str), source (str),
                                binary_label (int: 1=spam, 0=ham)
    """
    enron = _load_enron()
    sms   = _load_sms()
    df    = pd.concat([enron, sms], ignore_index=True)

    # Drop records with missing / too-short text
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() >= min_text_len].copy()

    # Normalise label column
    df["label"] = df["label"].str.lower().str.strip()

    # Deduplicate on exact text match
    before = len(df)
    if deduplicate:
        df = df.drop_duplicates(subset=["text"]).copy()
    logger.info(
        f"Combined dataset: {before} → {len(df)} after deduplication. "
        f"Spam: {(df['label']=='spam').sum()}, Ham: {(df['label']=='ham').sum()}"
    )

    # Binary label
    df["binary_label"] = (df["label"] == "spam").astype(int)

    df = df.reset_index(drop=True)
    return df


def train_test_split_stratified(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 split preserving the spam/ham ratio.
    """
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["binary_label"],
        random_state=random_state,
    )
    logger.info(
        f"Train: {len(train)} | Test: {len(test)} | "
        f"Train spam %: {train['binary_label'].mean()*100:.2f}% | "
        f"Test spam %:  {test['binary_label'].mean()*100:.2f}%"
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
