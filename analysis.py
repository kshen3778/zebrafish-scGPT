import os
print(os.getcwd())
import sys
print(sys.path)

import pandas as pd
import scanpy as sc
import anndata as ad
print("hi")
with open("data/exprMatrix.tsv") as your_data:
    adata = ad.read_csv(your_data, delimiter='\t')
    df = adata.to_df()
    print(df.head())
    for col in df.columns:
        print(col)
    print("rows")
    for row in df.index:
        print(row)
    print(len(df.index))