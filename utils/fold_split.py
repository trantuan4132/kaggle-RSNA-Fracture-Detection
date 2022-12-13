import argparse
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='train.csv',
                        help='Path to the label file')
    parser.add_argument('--label_cols', type=str, nargs='+',
                        default=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], 
                        help='Label columns used for fold splitting')
    parser.add_argument('--multi_label', action='store_true',
                        help='Whether label is multilabel or multiclass')
    parser.add_argument('--kfold', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--save_path', type=str, default='train_fold5.csv',
                        help='Path to the save file')
    return parser.parse_args()


def fold_split(df, kfold=5, y=None, groups=None):
    """
    Split the dataset into k folds

    Args:
    -----
    df: pandas dataframe
        Dataframe containing the labels
    kfold: int, default 5
        Number of folds
    y: pandas series, optional
        Target column used for creating stratified folds by preserving the percentage of samples for each class
    groups: pandas series, optional
        Group labels used for putting all samples of the same group into one fold
    """
    if groups is not None:
        kf = GroupKFold(n_splits=kfold)
        cv_split = kf.split(df.index, y=y, groups=groups)
    elif y is not None:
        kf = StratifiedKFold(n_splits=kfold, random_state=0)
        cv_split = kf.split(df.index, y=y)
    else:
        kf = KFold(n_splits=kfold, random_state=0)
        cv_split = kf.split(df.index)
        
    df['fold'] = -1
    for fold, (_, val_idx) in enumerate(cv_split):
        df.loc[val_idx, 'fold'] = fold
    return df


def main():
    args = parse_args()
    df = pd.read_pickle(args.label_path) if args.label_path.endswith('.pkl') else pd.read_csv(args.label_path)
    if args.multi_label:
        combined_label = (df[args.label_cols] > 0).astype('int').astype('str').values.sum(axis=1)
        y = pd.Series(combined_label).apply(lambda x: int(x, 2))
    else:
        y = df[args.label_cols]
    groups = df['StudyInstanceUID']

    df = fold_split(df, args.kfold, y=y, groups=groups)
    # save_path = args.label_path.replace('.csv', f'_fold{args.kfold}.csv')
    df.to_pickle(args.save_path) if args.label_path.endswith('.pkl') else df.to_csv(args.save_path, index=False)
    print(df)

    # # Plot fold distribution
    # plt.figure(figsize=(8, 6))
    # sns.countplot(x='fold', data=df)
    # plt.show()

    # # Plot fold distribution per label
    # label_cnt_per_fold = df.groupby('fold')[label_cols].sum().stack().reset_index()
    # label_cnt_per_fold.columns = ['fold', 'label', 'count']
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='count', y='label', hue='fold', 
    #             data=label_cnt_per_fold.sort_values('count', ascending=False))
    # plt.show()

if __name__ == '__main__':
    main()