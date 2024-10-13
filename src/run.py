import argparse

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
parser_simulate.add_argument('--moves_dir', type=str, required=True)
parser_simulate.add_argument('--output_dir', type=str, required=True)
parser_simulate.add_argument('--debug_mode', action='store_true')    

# Parse the arguments
args = parser.parse_args()

if args.command == 'generate_moves':
    # TODO: This is broken right now, and we'll need to fix it up before we next run
    # move generation.
    import move_generation
    move_generation.main(args)


elif args.command == 'simulate':
    # Load constants before everything else, since other packages 
    # expect these constants to be available at import-time.
    import constants 
    constants.load(args.moves_dir, args.debug_mode)

    import simulation
    simulation.run(args.output_dir)


else:
    parser.print_help()