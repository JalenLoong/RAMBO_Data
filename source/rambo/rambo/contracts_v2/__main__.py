"""Read-only validation CLI. Output is a report, never a publication action."""
import argparse
import json
from .validation import ContractError, validate_episode, validate_release

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('path')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--release-eligible',action='store_true')
    mode.add_argument('--release-manifest',action='store_true')
    args=p.parse_args()
    try:
        result=validate_release(args.path) if args.release_manifest else validate_episode(args.path,release=args.release_eligible)
    except ContractError as e:
        print(json.dumps({'passed':False,'code':e.code,'detail':e.detail}));return 1
    print(json.dumps({'passed':True,'result':result},sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
