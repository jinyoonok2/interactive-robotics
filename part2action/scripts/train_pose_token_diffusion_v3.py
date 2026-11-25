"""Explicit v3 entry point with diagnostics active before importing PyTorch."""
import argparse
import os

from diagnostics.run_health import RunHealth
from diagnostics.storage import configure_cache_environment, result_dir


def guarded_main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out-dir', required=True)
    args, _ = parser.parse_known_args()
    configure_cache_environment()
    output = result_dir(args.out_dir)
    directory = os.environ.get('P2A_JOB_HEALTH_DIR')
    directory = result_dir(directory) if directory else output.parent/'health'/f'{output.name}_{os.getpid()}'
    with RunHealth(directory/'python') as health:
        health.event('importing_trainer')
        from models.pose_token_diffusion_v3.training import main
        main(health=health)

if __name__ == '__main__':
    guarded_main()
