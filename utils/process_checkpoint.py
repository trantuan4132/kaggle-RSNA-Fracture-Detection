import torch
import timm
import argparse
import importlib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='convnext_tiny_22k_1k_384.pth',
                        help='Path to the checkpoint file')
    parser.add_argument('--model', type=str, default='convnext',
                        help='Model name')
    parser.add_argument('--variant', type=str, default='convnext_tiny',
                        help='Model variant')
    return parser.parse_args()


def preprocess_checkpoint(checkpoint_path, model, variant):
    try:
        mod = importlib.import_module(f'timm.models.{model}')
        if hasattr(mod, 'checkpoint_filter_fn'):
            model = timm.create_model(variant, pretrained=False)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint = mod.checkpoint_filter_fn(checkpoint, model)
            new_path = checkpoint_path.replace('.pth', '_altered.pth')
            torch.save(checkpoint, new_path)
            print("Preprocessed checkpoint saved to", new_path)
        else:
            print("No preprocessing function found or no need to preprocess")
    except ModuleNotFoundError:
        print(f'Model {model} not found')


if __name__ == "__main__":
    args = parse_args()
    preprocess_checkpoint(args.checkpoint_path, args.model, args.variant)