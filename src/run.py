import argparse
import logging
import os

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Blockus scripts entrypoint")

    # Create subparsers for the two commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for 'generate_moves'
    parser_generate = subparsers.add_parser('generate_moves', help='Pre-generate move data')
    parser_generate.add_argument('--board_size', type=int, required=True)
    parser_generate.add_argument('--output_dir', type=str, required=True)

    # Subparser for 'simulate'
    parser_simulate = subparsers.add_parser('simulate', help='Run self-play sessions')
    parser_simulate.add_argument('--config', type=str, required=True)
    parser_simulate.add_argument('--output_dir', type=str, required=True)

    # Subparser for 'serve'
    parser_simulate = subparsers.add_parser('serve', help='Run model server')
    parser_simulate.add_argument('--config', type=str, required=True)

    # Subparser for 'train'
    parser_train = subparsers.add_parser('train', help='Train neural network')
    parser_train.add_argument('--moves_dir', type=str, required=True)
    parser_train.add_argument('--games_dir', type=str, required=True)
    parser_train.add_argument('--output_dir', type=str, required=True)

    # Parse the arguments
    args = parser.parse_args()

    if args.command == 'generate_moves':
        # TODO: This is broken right now, and we'll need to fix it up before we next run
        # move generation.
        import move_generation
        move_generation.main(args)


    elif args.command == 'simulate':
        # Bit hacky, but we store the config path in an environment variable so
        # that this process and all children processes can access it as needed to
        # load the config.
        os.environ["CONFIG_PATH"] = args.config

        import simulation
        simulation.run()


    elif args.command == 'serve':
        # Bit hacky, but we store the config path in an environment variable so
        # that this process and all children processes can access it as needed to
        # load the config.
        os.environ["CONFIG_PATH"] = args.config

        import inference.server
        inference.server.run()


    elif args.command == 'train':
        import config as config 
        config.load(args.moves_dir, False)

        import training
        training.run(args.games_dir, args.output_dir)


    else:
        parser.print_help()


if __name__ == "__main__":
    main()