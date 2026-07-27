import pandas as pd


datasets = [
    "pcos_dataset.csv",
    "pcos_new_dataset.csv"
]


for file in datasets:

    print("\n====================")
    print(file)
    print("====================")


    df = pd.read_csv(file)


    print("\nShape:")
    print(df.shape)


    print("\nColumns:")
    print(df.columns.tolist())


    print("\nMissing Values:")
    print(df.isna().sum())


    print("\nPreview:")
    print(df.head())