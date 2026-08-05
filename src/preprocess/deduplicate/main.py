import argparse

from .exact import run as exact_run
from .near import run as near_run


parser = argparse.ArgumentParser()

parser.add_argument(
    "mode",
    choices=[
        "exact",
        "near",
        "all",
    ]
)

args = parser.parse_args()


if args.mode == "exact":

    exact_run()

elif args.mode == "near":

    near_run()

elif args.mode == "all":

    exact_run()
    near_run()