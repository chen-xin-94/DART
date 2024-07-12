import subprocess
import argparse

parser = argparse.ArgumentParser(description="Run a command")
parser.add_argument(
    "-c", "--command_path", type=str, help="Absolute path to the command to run"
)


def run_command(command_path):

    # Make the script executable
    subprocess.run(["chmod", "+x", str(command_path)], check=True)

    # Run the script
    print(f"Running command: {command_path}")
    result = subprocess.run(["bash", str(command_path)], capture_output=True, text=True)

    return result.stdout, result.stderr


if __name__ == "__main__":
    args = parser.parse_args()
    stdout, stderr = run_command(args.command_path)

    print("STDOUT:")
    print(stdout)

    if stderr:
        print("STDERR:")
        print(stderr)
