# compare-recommandation-algorithm

MovieLens recommender comparison under controlled observed-rating sparsity.

## Files

- `Method1_UserUserCF.ipynb`: existing user-user CF notebook.
- `method1_user_user_cf.py`: Method 1, user-user CF with Pearson similarity.
- `data_pipeline.py`: shared MovieLens loading, masking, splitting, matrix building, and RMSE evaluation.
- `method2_item_item_cf.py`: Method 2, item-item CF with cosine similarity.
- `method3_latent_factor_sgd.py`: Method 3, biased latent factor model trained with SGD.
- `compare_results.py`: combines method CSVs, creates final RMSE/runtime plots, and reports breakdown thresholds.

## Example runs

```bash
python3 -m pip install -r requirements.txt

mkdir -p data
curl -L https://files.grouplens.org/datasets/movielens/ml-1m.zip -o data/ml-1m.zip
unzip data/ml-1m.zip -d data

python3 method1_user_user_cf.py data/ml-1m/ratings.dat
python3 method2_item_item_cf.py data/ml-1m/ratings.dat
python3 method3_latent_factor_sgd.py data/ml-1m/ratings.dat
python3 compare_results.py results/method1_user_user_cf.csv results/method2_item_item_cf.csv results/method3_latent_factor_sgd.csv
```

The sparsity levels remove a controlled fraction of the observed MovieLens
ratings. The CSV also reports `actual_matrix_missing`, because MovieLens is
already naturally sparse before artificial masking.

Each method CSV includes the same-split global mean baseline, `rmse_vs_baseline`,
`beats_baseline`, skipped-test-pair rate, and neighbourhood diagnostics where
applicable. `compare_results.py` writes `results/breakdown_summary.csv` with the
first sparsity level where each method no longer beats the baseline.
