#!/usr/bin/env python3
"""
Randomise animal positions in the Gazebo / Ignition world "large_demo".

- Moves existing models; it does NOT spawn duplicate models.
- Avoids known trees, rocks, tent, campfire, gazebo structure, etc.
- Keeps animals separated from each other.
- Randomises yaw as well as x / y.
- Automatically supports both:
    * Ignition Gazebo / Fortress: `ign service`, ignition.msgs.*
    * New Gazebo Sim:            `gz service`,  gz.msgs.*

Test:
    python3 random_animal_spawn.py

Reproducible layout:
    python3 random_animal_spawn.py --seed 123

Only show selected positions:
    python3 random_animal_spawn.py --dry-run
"""

import argparse
import math
import random
import shutil
import subprocess
import sys
import time


WORLD_NAME = "large_demo"

# Stay away from the walls, which are close to +/-12.5 m in large_demo.sdf.
X_MIN, X_MAX = -10.0, 10.0
Y_MIN, Y_MAX = -10.0, 10.0

# Minimum centre-to-centre separation between animals.
MIN_ANIMAL_DISTANCE = 2.5

# Extra padding added around every obstacle exclusion radius.
EXTRA_OBSTACLE_MARGIN = 0.35

# Maximum attempts to find a valid position for one animal.
MAX_ATTEMPTS = 5000

# Existing animal model names in large_demo.sdf.
# z can be adjusted here if any mesh sits too high / low.
ANIMALS = {
    "female_deer1": 0.0,
    "male_deer1": 0.0,
    "kangaroo1": 0.0,
    "boar1": 0.0,
}

# Known obstacle centres from large_demo.sdf.
# Format: (name, x, y, exclusion_radius_metres)
#
# These radii are deliberately conservative because many of the external
# Fuel models do not expose their full geometry directly in the world SDF.
OBSTACLES = [
    # Oak trees
    ("oak1",  -4.77393,  12.2995, 1.35),
    ("oak2",   2.99444,  -4.95498, 1.35),
    ("oak3",  -9.12426, -11.2689, 1.35),
    ("oak4",   4.21788,  -1.41417, 1.35),
    ("oak5",  -6.26553,   5.00575, 1.35),
    ("oak6",  -4.33701,   1.32854, 1.35),
    ("oak7",  -0.620372, 10.0, 1.35),
    ("oak8",   3.54698,   5.0, 1.35),

    # Pine trees
    ("pine1",   6.05721,   9.78933, 1.25),
    ("pine2", -10.0184,    8.85188, 1.25),
    ("pine3",  -4.62343, -12.5618, 1.25),
    ("pine4",   7.08738,   1.33612, 1.25),
    ("pine5",  -4.64898,  -3.0329, 1.25),
    ("pine6",  -2.27195,  -8.68441, 1.25),

    # Other existing objects
    ("platypus1", -8.0,  6.0, 1.0),
    ("gazebo1",   10.0, -10.0, 3.0),

    # Rocks
    ("rock1",  -6.30926, -8.38159, 1.25),
    ("rock2",  -8.27162, -6.19647, 1.25),
    ("rock3",  -9.2841,  -4.61803, 1.25),
    ("rock4", -10.0,    -12.0,     1.25),
    ("rock5", -12.0,    -12.0,     1.25),
    ("rock6", -14.0,    -12.0,     1.25),

    # Scenario props
    ("tent1",     -9.0, -6.0, 2.0),
    ("campfire1", -7.0, -6.0, 1.4),

    # Dynamic-demo objects that are initially visible in this world
    ("demo_animal",       -4.0,  4.0, 1.0),
    ("demo_tree_healthy",  4.0, -4.0, 1.25),

    # Optional reserved spawn point for Husky.
    # If your Husky starts somewhere else, edit or remove this line.
    ("husky_start_reserved", 0.0, 0.0, 2.0),
]


def distance_2d(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def clear_of_obstacles(x, y):
    """Return (True, None) if point is outside all obstacle exclusion zones."""
    for name, ox, oy, radius in OBSTACLES:
        required = radius + EXTRA_OBSTACLE_MARGIN
        if distance_2d(x, y, ox, oy) < required:
            return False, name
    return True, None


def clear_of_animals(x, y, chosen_positions):
    """Avoid overlap / unrealistic clustering between randomly placed animals."""
    for other_name, (ox, oy, _yaw) in chosen_positions.items():
        if distance_2d(x, y, ox, oy) < MIN_ANIMAL_DISTANCE:
            return False, other_name
    return True, None


def choose_positions(rng):
    """Choose a collision-avoiding random x, y and yaw for every animal."""
    chosen = {}

    for animal_name in ANIMALS:
        for _ in range(MAX_ATTEMPTS):
            x = rng.uniform(X_MIN, X_MAX)
            y = rng.uniform(Y_MIN, Y_MAX)
            yaw = rng.uniform(-math.pi, math.pi)

            obstacle_ok, _ = clear_of_obstacles(x, y)
            animals_ok, _ = clear_of_animals(x, y, chosen)

            if obstacle_ok and animals_ok:
                chosen[animal_name] = (x, y, yaw)
                break
        else:
            raise RuntimeError(
                f"Could not find a valid position for {animal_name} after "
                f"{MAX_ATTEMPTS} attempts. Reduce safety distances or enlarge "
                f"the spawn region."
            )

    return chosen


def detect_gazebo_cli():
    if shutil.which("ign"):
        return "ign", "ignition.msgs"

    if shutil.which("gz"):
        return "gz", "gz.msgs"

    raise RuntimeError(
        "Neither `ign` nor `gz` was found in PATH."
    )


def wait_for_set_pose_service(cli, timeout_seconds=20.0):
    """Wait until Gazebo has created /world/large_demo/set_pose."""
    target = f"/world/{WORLD_NAME}/set_pose"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            result = subprocess.run(
                [cli, "service", "--list"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if target in result.stdout:
                return True
        except (subprocess.SubprocessError, OSError):
            pass

        time.sleep(0.5)

    return False


def set_pose(cli, msg_ns, model_name, x, y, z, yaw):
    """Move one existing Gazebo model using the UserCommands set_pose service."""
    service = f"/world/{WORLD_NAME}/set_pose"

    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    request = (
        f'name: "{model_name}", '
        f'position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}}, '
        f'orientation: {{x: 0.0, y: 0.0, z: {qz:.9f}, w: {qw:.9f}}}'
    )

    cmd = [
        cli,
        "service",
        "-s", service,
        "--reqtype", f"{msg_ns}.Pose",
        "--reptype", f"{msg_ns}.Boolean",
        "--timeout", "3000",
        "--req", request,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    combined = (result.stdout + "\n" + result.stderr).strip()

    # Different Gazebo / Ignition versions format their output slightly
    # differently, so return code is the main check.
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to move {model_name}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{combined}"
        )

    return combined


def print_positions(positions):
    print("\nSelected random animal positions:")
    print("-" * 67)
    print(f"{'animal':18s} {'x (m)':>10s} {'y (m)':>10s} {'yaw (deg)':>12s}")
    print("-" * 67)

    for name, (x, y, yaw) in positions.items():
        print(
            f"{name:18s} "
            f"{x:10.3f} "
            f"{y:10.3f} "
            f"{math.degrees(yaw):12.1f}"
        )

    print("-" * 67)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Use a fixed seed when you want the same layout again.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Choose and print positions but do not move Gazebo models.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    positions = choose_positions(rng)
    print_positions(positions)

    if args.dry_run:
        print("\nDry run only: Gazebo was not changed.")
        return 0

    try:
        cli, msg_ns = detect_gazebo_cli()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nUsing Gazebo command: {cli}")
    print(f"Message namespace: {msg_ns}")
    print(f"Waiting for /world/{WORLD_NAME}/set_pose ...")

    if not wait_for_set_pose_service(cli):
        print(
            "\nERROR: set_pose service was not found.\n"
            f"Check with:\n    {cli} service --list | grep "
            f"/world/{WORLD_NAME}/set_pose\n\n"
            "If it is missing, the UserCommands system plugin may need to be "
            "added to the world.",
            file=sys.stderr,
        )
        return 1

    print("Service found. Moving animals...")

    for animal_name, (x, y, yaw) in positions.items():
        z = ANIMALS[animal_name]

        try:
            set_pose(
                cli=cli,
                msg_ns=msg_ns,
                model_name=animal_name,
                x=x,
                y=y,
                z=z,
                yaw=yaw,
            )
        except RuntimeError as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            return 1

        print(
            f"  OK  {animal_name:18s} -> "
            f"x={x:6.2f}, y={y:6.2f}, yaw={math.degrees(yaw):6.1f} deg"
        )

    print("\nDone. All animals were randomised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())