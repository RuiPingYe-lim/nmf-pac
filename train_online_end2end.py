#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json
from uda_core.config import build_parser
from uda_core.trainer import train


if __name__ == '__main__':
    args = build_parser().parse_args()
    print(f'Command line arguments: {vars(args)}') 
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    train(args) 