import pandas as pd

# CHECK 1: Target shift
print('=== CHECK 1: Target shifted correctly? ===')
tech = pd.read_csv('data/processed/technical_features.csv', index_col=0, parse_dates=True)
close = pd.read_csv('data/raw/spy_daily.csv', index_col=0, parse_dates=True)['Close']
next_day_up = (close.pct_change().shift(-1) > 0).astype(int)
comparison = tech['target'] == next_day_up.reindex(tech.index)
mismatches = (~comparison).sum()
print(f'Mismatches: {mismatches}')
print('PASS' if mismatches == 0 else 'FAIL — target may not be shifted correctly!')

print()

# CHECK 2: PCA fit on train only
print('=== CHECK 2: PCA fit on train only? ===')
final = pd.read_csv('data/processed/final_features.csv', index_col=0, parse_dates=True)
pca_cols = [c for c in final.columns if c.startswith('news_pca_')]
train_pca = final.loc[final.index < '2021-01-01', pca_cols]
test_pca = final.loc[final.index >= '2022-07-01', pca_cols]
train_mean = train_pca.mean().abs().mean()
test_mean = test_pca.mean().abs().mean()
print(f'Train PCA mean of abs means: {train_mean:.6f} (should be ~0)')
print(f'Test PCA mean of abs means:  {test_mean:.6f} (should differ from 0)')
print('PASS' if train_mean < 0.001 and test_mean > 0.001 else 'FAIL — PCA may have been fit on all data!')
