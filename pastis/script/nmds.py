#! /usr/bin/env python

from pastis.algorithms import run_nmds


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run NMDS.')
    parser.add_argument('directory', type=str,
                        help='directory', default=None)
    args = parser.parse_args()

    if args.directory is not None and not "":
        run_nmds(args.directory)


if __name__ == "__main__":
    main()
