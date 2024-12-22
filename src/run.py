import argparse
import logging
import os

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Blockus scripts entrypoint")

    # Create subparsers for the two commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for 'simulate'
    parser_simulate = subparsers.add_parser('simulate', help='Run self-play sessions')
    parser_simulate.add_argument('--config', type=str, nargs='+', required=True, help='Paths to one or more config files')
    parser_simulate.add_argument('--config-override', type=str, nargs='*', required=False, help='Individual config values to override')

    # Subparser for 'server'
    parser_server = subparsers.add_parser('server', help='Run gameplay server')
    parser_server.add_argument('--config', type=str, nargs='+', required=True, help='Paths to one or more config files')
    parser_server.add_argument('--config-override', type=str, nargs='*', required=False, help='Individual config values to override')

    # Subparser for 'training_unlooped'
    parser_training_unlooped = subparsers.add_parser('training_unlooped', help='Run training for development purposes')
    parser_training_unlooped.add_argument('--config', type=str, nargs='+', required=True, help='Paths to one or more config files')    
    parser_training_unlooped.add_argument('--config-override', type=str, nargs='*', required=False, help='Individual config values to override')

    # Parse the arguments
    args = parser.parse_args()

    if args.command == 'simulate':
        # Bit hacky, but we store the config path in an environment variable so
        # that this process and all children processes can access it as needed to
        # load the config.
        os.environ["CONFIG_PATHS"] = ",".join(args.config)
        if args.config_override:
            os.environ["CONFIG_OVERRIDES"] = ",".join(args.config_override)

        import simulation
        simulation.run()
    
    elif args.command == 'server':
        # Bit hacky, but we store the config path in an environment variable so
        # that this process and all children processes can access it as needed to
        # load the config.
        os.environ["CONFIG_PATHS"] = ",".join(args.config)
        if args.config_override:
            os.environ["CONFIG_OVERRIDES"] = ",".join(args.config_override)        

        import server
        server.run()

    elif args.command == 'training_unlooped':
        os.environ["CONFIG_PATHS"] = ",".join(args.config)
        if args.config_override:
            os.environ["CONFIG_OVERRIDES"] = ",".join(args.config_override)

        import training.unlooped
        training.unlooped.run()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()