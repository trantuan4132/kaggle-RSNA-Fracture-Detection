import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import rc
rc('animation', html='jshtml')
import seaborn as sns
import cv2


def square_bbox(x0, y0, x1, y1, low=0, high=1):
    w, h = x1 - x0, y1 - y0
    w_pad, h_pad = h - min(h, w), w - min(h, w)
    return np.array([x0-w_pad//2, y0-h_pad//2, x1+w_pad//2, y1+h_pad//2]).clip(low, high)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_file_1', type=str, default='output/infer_stage_1.pkl', 
                        help="Path to the label file of the 1st stage")
    parser.add_argument('--label_file_2', type=str, default='output/infer_stage_2.csv', 
                        help="Path to the label file of the 2nd stage")
    parser.add_argument('--image_dir', type=str, default='dataset/train_images', 
                        help="Path to the image directory")
    parser.add_argument('--image_format', type=str, default='dcm', 
                        help="The extension of image file")
    parser.add_argument('--start_idx', type=int, default=None, 
                        help="Index of the first slice to be displayed")
    parser.add_argument('--end_idx', type=int, default=None, 
                        help="Index of the last slice to be displayed")
    parser.add_argument('--n_steps', type=int, default=2, 
                        help="Number of steps after which slices are selected to be displayed")
    parser.add_argument('--out_file', type=str, default='output/demo.mp4', 
                        help="Path to the output file")
    parser.add_argument('--fps', type=int, default=10, 
                        help="The FPS of the animation when output file is saved")
    args = parser.parse_args()

    img_cols = ['StudyInstanceUID', 'Slice']
    label_cols = ['x0', 'y0', 'x1', 'y1', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    # Load data
    df = pd.read_pickle(args.label_file_1) if args.label_file_1.endswith('pkl') else pd.read_csv(args.label_file_1)

    # Read 1st uid group
    df = df.groupby(img_cols[0])
    df = df.get_group((list(df.groups)[0]))

    # Select slices to be displayed
    df = df.iloc[args.start_idx : args.end_idx : args.n_steps]

    # Merge two label files
    df = df.merge(pd.read_csv(args.label_file_2), on=img_cols[0], suffixes=(None, '_frac'))
    frac_cols = [col + '_frac' for col in label_cols[4:]]
    df[frac_cols] = (df[frac_cols] * (df[label_cols[4:]] > 0.5).values).cummax()
    
    # Get bounding box coordination
    xmin, ymin = df[(df[label_cols[4:]] > 0.5).any(axis=1)][label_cols[0:2]].min()
    xmax, ymax = df[(df[label_cols[4:]] > 0.5).any(axis=1)][label_cols[2:4]].max()

    # Initialize the plot
    fig, axes = plt.subplots(2, 2, figsize=(14,7))
    axes[0, 0].text(-0.8, 0.5, 'Stage 1', fontsize=20, transform=axes[0, 0].transAxes)
    axes[1, 0].text(-0.8, 0.5, 'Stage 2', fontsize=20, transform=axes[1, 0].transAxes)

    def animate(i):
        row = df.iloc[i]
        img = cv2.imread(f'{args.image_dir}/{row[img_cols[0]]}/{row[img_cols[1]]}.{args.image_format}')
        # img = np.zeros((512, 512, 3))
        x0, y0, x1, y1 = map(lambda x: int(x*img.shape[0]), [xmin, ymin, xmax, ymax])
        x0, y0, x1, y1 = square_bbox(x0, y0, x1, y1, low=0, high=img.shape[0])
        crop_img = img[y0:y1, x0:x1].copy()
        axes[0, 0].imshow(cv2.rectangle(img, (x0, y0), (x1, y1), (255,0,0), 3).astype(int), cmap='bone')
        axes[0, 0].axis('off')

        sns.heatmap(pd.DataFrame(row[label_cols[4:]].astype('float')), 
                    vmin=0.0, vmax=1.0, cbar=False, cmap='Blues', ax=axes[0, 1])
        axes[0, 1].tick_params(axis='y', labelrotation=0)
        axes[0, 1].axes.get_xaxis().set_visible(False)
        axes[0, 1].set_title('Vertebrae')

        axes[1, 0].imshow(crop_img.astype(int), cmap='bone')
        axes[1, 0].axis('off')

        sns.heatmap(pd.DataFrame(row[frac_cols].values.astype('float'), index=label_cols[4:]), 
                    vmin=-1.0, vmax=1.0, cbar=False, cmap='bwr', ax=axes[1, 1])
        axes[1, 1].tick_params(axis='y', labelrotation=0)
        axes[1, 1].axes.get_xaxis().set_visible(False)
        axes[1, 1].set_title('Fracture')

    # Create animation
    ani = FuncAnimation(fig, animate, frames=len(df),
                        interval=500, repeat=False, blit=False)
    # plt.show()
    plt.close()

    # Save animation
    ani.save(args.out_file, fps=args.fps)

if __name__ == '__main__':
    main()