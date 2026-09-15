import argparse
import json
from .validation import DataPaths, read_json, validate_canonical, validate_raw, ContractError

def main():
    parser=argparse.ArgumentParser(description='Read-only wam-quadruped-v2.1.0 data validation')
    parser.add_argument('--config',required=True)
    parser.add_argument('--raw',action='store_true')
    args=parser.parse_args()
    try:
        paths=DataPaths.from_config(read_json(args.config))
        result=validate_raw(paths.dataset_root,tools=paths.tools) if args.raw else validate_canonical(paths.canonical_root,tools=paths.tools)
    except ContractError as error:
        print(json.dumps({'passed':False,'code':error.code,'detail':error.detail}));return 1
    print(json.dumps({'passed':True,'result':result},ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
