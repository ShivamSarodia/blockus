import argparse
import move_generation
import simulation


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
    parser_simulate.add_argument('--moves_dir', type=str, required=True)
    parser_simulate.add_argument('--output_dir', type=str, required=True)
    parser_simulate.add_argument('--debug_mode', action='store_true')    

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    if args.command == 'generate_moves':
        move_generation.main(args)
    elif args.command == 'simulate':
        simulation.main(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()