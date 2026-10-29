"""One-off stratified split of the raw dataset into a training file and the held-out test file."""
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def main():
    df = pd.read_csv(config.DATA_PATH)
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET_COL],
        random_state=config.RANDOM_STATE,
    )
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    for name, part in [("train", train_df), ("test", test_df)]:
        share = part[config.TARGET_COL].mean()
        print(f"{name}: {len(part)} rows, churn share {share:.4f}")


if __name__ == "__main__":
    main()
