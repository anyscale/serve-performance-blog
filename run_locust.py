import argparse
import os
import subprocess

DEFAULT_CONCURRENCIES = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500]
# DEFAULT_CONCURRENCIES = [400]


def run_one_load_test(
    users: int,
    run_time: int,
    token: str,
    host: str,
    route_prefix: str,
    name: str,
    output_dir: str,
    locustfile: str,
    processes: int,
):
    run_dir = os.path.join(output_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    json_filename = f"{run_dir}/{name}_{users}"
    cmd = [
        "locust",
        "--headless",
        "-f", locustfile,
        "-u", str(users),
        "-r", str(50),
        "-t", str(run_time),
        "--processes", str(processes),
        "--host", host,
        "--route_prefix", route_prefix,
        "--json-file", json_filename,
        "--reset-stats",
    ]

    if token:
        cmd += [f"--token={token}"]

    print("Starting load test: " + " ".join(cmd))
    subprocess.run(cmd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--run-time", default=60, type=int,
        help="Test duration. Same option as Locust's --run-time.",
    )
    parser.add_argument(
        "--token", default="", type=str,
        help="Bearer token to use when querying the service.",
    )
    parser.add_argument(
        "--host", type=str, required=True,
        help="Hostname to use when querying the service.",
    )
    parser.add_argument(
        "--route-prefix", type=str, default="/",
        help="Route prefix to use when querying the service.",
    )
    parser.add_argument(
        "-n", "--name", type=str, default="locust",
        help="Name of the load test.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="results",
        help="Directory to write JSON result files to.",
    )
    parser.add_argument(
        "-f", "--locustfile", type=str, default="locustfile.py",
        help="Path to the Locust file.",
    )
    parser.add_argument(
        "-p", "--processes", type=int, default=4,
        help="Number of Locust worker processes.",
    )
    parser.add_argument(
        "--concurrencies", type=str, default="",
        help="Comma-separated concurrency levels (default: 10-500).",
    )
    return parser.parse_args()


def main(args):
    if args.concurrencies:
        concurrencies = [int(c) for c in args.concurrencies.split(",")]
    else:
        concurrencies = DEFAULT_CONCURRENCIES

    for users in concurrencies:
        run_one_load_test(
            users, args.run_time, args.token, args.host,
            args.route_prefix, args.name, args.output_dir,
            args.locustfile, args.processes,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
