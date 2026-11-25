"""Separate entry point for Pose-token Diffusion; original trainers stay frozen."""
from _common import ROOT  # adds part2action to sys.path
from models.pose_token_diffusion.training import main

if __name__ == '__main__':
    main()
