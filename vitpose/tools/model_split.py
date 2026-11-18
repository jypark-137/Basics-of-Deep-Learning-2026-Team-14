"""Split a ViTPose++ checkpoint into root checkpoints/split/.

Run from the project root: python vitpose/tools/model_split.py
--source and --target accept absolute paths or paths relative to that root.
"""

import torch
import os
import argparse
import copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', type=Path,
                        default=Path('checkpoints/vitpose-plus-small.pth'),
                        help='Input ViTPose++ checkpoint')
    parser.add_argument('--target', type=Path, default=Path('checkpoints/split'),
                        help='Output directory for task checkpoints')
    args = parser.parse_args(argv)
    args.source = (PROJECT_ROOT / args.source.expanduser()).resolve()
    args.target = (PROJECT_ROOT / args.target.expanduser()).resolve()
    return args


def main(argv=None):

    args = parse_args(argv)

    if not args.source.is_file():
        raise FileNotFoundError(f'Input checkpoint not found: {args.source}')

    ckpt = torch.load(args.source, map_location='cpu')
    args.target.mkdir(parents=True, exist_ok=True)
    print('Source:', args.source)
    print('Output directory:', args.target)
    
    experts = dict()

    new_ckpt = copy.deepcopy(ckpt)

    state_dict = new_ckpt['state_dict']

    for key, value in state_dict.items():
        if 'mlp.experts' in key:
            experts[key] = value

    keys = ckpt['state_dict'].keys()

    target_expert = 0
    new_ckpt = copy.deepcopy(ckpt)

    for key in keys:
        if 'mlp.fc2' in key:
            value = new_ckpt['state_dict'][key]
            value = torch.cat([value, experts[key.replace('fc2.', f'experts.{target_expert}.')]], dim=0)
            new_ckpt['state_dict'][key] = value

    torch.save(new_ckpt, os.path.join(args.target, 'coco.pth'))

    names = ['aic', 'mpii', 'ap10k', 'apt36k','wholebody']
    num_keypoints = [14, 16, 17, 17, 133]
    weight_names = ['keypoint_head.deconv_layers.0.weight', 
                    'keypoint_head.deconv_layers.1.weight', 
                    'keypoint_head.deconv_layers.1.bias', 
                    'keypoint_head.deconv_layers.1.running_mean', 
                    'keypoint_head.deconv_layers.1.running_var', 
                    'keypoint_head.deconv_layers.1.num_batches_tracked', 
                    'keypoint_head.deconv_layers.3.weight', 
                    'keypoint_head.deconv_layers.4.weight', 
                    'keypoint_head.deconv_layers.4.bias', 
                    'keypoint_head.deconv_layers.4.running_mean', 
                    'keypoint_head.deconv_layers.4.running_var', 
                    'keypoint_head.deconv_layers.4.num_batches_tracked', 
                    'keypoint_head.final_layer.weight', 
                    'keypoint_head.final_layer.bias']
    
    exist_range = True

    for i in range(5):

        new_ckpt = copy.deepcopy(ckpt)

        target_expert = i + 1

        for key in keys:
            if 'mlp.fc2' in key:
                expert_key = key.replace('fc2.', f'experts.{target_expert}.')
                if expert_key in experts:
                    value = new_ckpt['state_dict'][key]
                    value = torch.cat([value, experts[expert_key]], dim=0)
                else:
                    exist_range = False

                new_ckpt['state_dict'][key] = value

        if not exist_range:
            break

        for tensor_name in weight_names:
            new_ckpt['state_dict'][tensor_name] = new_ckpt['state_dict'][tensor_name.replace('keypoint_head', f'associate_keypoint_heads.{i}')]

        for tensor_name in ['keypoint_head.final_layer.weight', 'keypoint_head.final_layer.bias']:
            new_ckpt['state_dict'][tensor_name] = new_ckpt['state_dict'][tensor_name][:num_keypoints[i]]
        
        # remove unnecessary part in the state dict
        for j in range(5):
            # remove associate part
            for tensor_name in weight_names:
                new_ckpt['state_dict'].pop(tensor_name.replace('keypoint_head', f'associate_keypoint_heads.{j}'))
        # remove expert part
        for key in list(new_ckpt['state_dict']):
            if 'mlp.experts.' in key:
                new_ckpt['state_dict'].pop(key)
            
        torch.save(new_ckpt, os.path.join(args.target, f'{names[i]}.pth'))

if __name__ == '__main__':
    main()
