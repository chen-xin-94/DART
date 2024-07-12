import subprocess
import argparse

from pathlib import Path
from datetime import datetime


def run_command(command_path):

    # Make the script executable
    subprocess.run(["chmod", "+x", str(command_path)], check=True)

    # Run the script
    print(f"Running: {command_path}")
    result = subprocess.run(["bash", str(command_path)], capture_output=True, text=True)

    return result.stdout, result.stderr


def main():
    parser = argparse.ArgumentParser(
        description="Run all commands recursively in a directory"
    )
    parser.add_argument(
        "--objs", type=str, help="List of objects to run in str", default=None
    )
    parser.add_argument(
        "--model_name", type=str, help="Name of the model", default="sdxl"
    )
    parser.add_argument(
        "--repo_dir",
        type=str,
        help="Directory of the repository",
        default="/mnt/ssd2/xin/repo/DART/diversification/dreambooth",
    )
    parser.add_argument(
        "--instance_dir",
        type=str,
        help="Directory of the instance data",
        default="/mnt/ssd2/xin/repo/DART/diversification/instance_data",
    )

    args = parser.parse_args()
    repo_dir = Path(args.repo_dir)
    instance_dir = Path(args.instance_dir)
    objs = args.objs

    if objs is None:
        objs_ = sorted([obj.name for obj in instance_dir.iterdir() if obj.is_dir()])
        objs = [obj_.replace("_", " ") for obj_ in objs_]
    else:
        objs = objs.split(", ")
    print(objs)

    # setup paths
    script_dir = repo_dir / "scripts" / args.model_name
    output_dir = repo_dir / "output" / args.model_name

    finished = []
    skipped = []
    failed = []
    errors = []
    for obj in objs:
        obj_script_dir = script_dir / obj.replace(" ", "_")

        for command_path in obj_script_dir.rglob("*.sh"):
            instance_variant = command_path.stem
            obj = command_path.parent.name
            output_path = output_dir / obj / instance_variant

            # check if the output exists

            if output_path.exists():
                print(f"Skipped {instance_variant} because output already exists")
                skipped.append(str(command_path))
                continue

            # run the command
            stdout, stderr = run_command(str(command_path))

            print("STDOUT:")
            print(stdout)

            if stderr:
                print("STDERR:")
                print(stderr)
                failed.append(str(command_path))
                errors.append(stderr)
                # # delete the output directory
                # subprocess.run(["rm", "-rf", str(output_path)], check=True)

    # if finished:
    #     print("Finished scripts:")
    #     print(finished)
    # if skipped:
    #     print("Skipped scripts:")
    #     print(skipped)
    # if failed:
    #     print("Failed scripts:")
    #     print(failed)

    # save the errors to a log file
    # add timestamp to the file name
    filename = f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(repo_dir / filename, "w") as f:
        for i in range(len(failed)):
            f.write(f"Script: {failed[i]}\n")
            f.write(f"Error: {errors[i]}\n\n")


if __name__ == "__main__":
    main()
