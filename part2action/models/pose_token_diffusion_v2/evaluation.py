"""V2 checkpoint entry point using the shared PartGym evaluation harness."""
def main():
    from models.pose_token_diffusion.evaluation import main as evaluate
    evaluate(architecture='pose_token_diffusion_v2')


if __name__ == '__main__':
    main()
