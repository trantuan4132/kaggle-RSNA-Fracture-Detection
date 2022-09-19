import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='best_score',
                        help='Choose mode for processing log file (best_score, visualize)')
    parser.add_argument('--log_file', type=str, default='teacher_checkpoint/log.txt',
                        help='Path to the log file')
    parser.add_argument('--metric', type=str, default='val_auc',
                        help='Metric to be considered')
    parser.add_argument('--return_avg', action='store_true',
                        help='Return average score')
    parser.add_argument('--title', type=str, default='Validation AUC',
                        help='Plot title for visualization')
    return parser.parse_args()


def log_file_to_df(log_file='log.txt'):
    """Convert data from log file to pandas dataframe"""
    with open(log_file) as fh:
        data = fh.read().replace('\n', ',')
        return pd.DataFrame(eval(f"[{data}]"))


def best_score_from_log_file(log_file='log.txt', metric='val_auc', return_avg=False):
    """Get best score from log file"""
    df = log_file_to_df(log_file)
    df = df[df.groupby(['fold'])[metric].transform(max)==df[metric]]
    if return_avg:
        return df[metric].mean()
    return df 


def visualize_log_file(log_file='log.txt', metric='val_auc', title='Validation AUC'):
    """Visualize data from log file"""
    df = log_file_to_df(log_file)
    if metric in df.columns:
        plt.figure(figsize=(9,6))
        sns.lineplot(x='epoch', y=metric, hue='fold', 
                    data=df.assign(fold=lambda x: x['fold'].astype('str')))
        plt.title(title)
    elif metric == 'loss':
        df = df.melt(['fold', 'epoch'], ['train_loss', 'val_loss'], var_name='loss')
        df['loss'] = df['loss'].str[:-5] + '_fold_' + df['fold'].astype('str')
        plt.figure(figsize=(9,6))
        sns.lineplot(x='epoch', y='value', hue='loss', data=df).legend_.set_title(None)
        plt.title('Training and Validation Loss')
    # plt.savefig('figure.png')
    plt.show()


def visualize_log_files(log_files={'name': 'log.txt'}, metric='val_auc', title='Validation AUC'):
    """Visualize data from dictionary containing pairs of method's name and log file"""
    dfs = [log_file_to_df(log_file).assign(method=method) for method, log_file in log_files.items()]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    if metric in df.columns:
        plt.figure(figsize=(9,6))
        sns.lineplot(x='epoch', y=metric, hue='method', 
                    data=df.assign(fold=lambda x: x['fold'].astype('str')))
        plt.title(title)
    # plt.savefig('figure.png')
    plt.show()


def main():
    args = parse_args()
    if args.mode == 'best_score':
        res = best_score_from_log_file(args.log_file, args.metric, args.return_avg)
        print(res)
    elif args.mode == 'visualize':
        visualize_log_file(args.log_file, args.metric, args.title)


if __name__ == "__main__":
    main()