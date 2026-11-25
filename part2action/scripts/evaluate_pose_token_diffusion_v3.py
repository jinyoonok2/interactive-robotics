"""V3-only PartGym entry point; does not redirect earlier evaluation runners."""
from models.pose_token_diffusion_v3.evaluation import main

if __name__ == '__main__':
    main()
