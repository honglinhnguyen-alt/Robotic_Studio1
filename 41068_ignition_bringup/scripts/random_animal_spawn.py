#!/usr/bin/env python3
"""
Randomise the animal layout of a running Gazebo world and write its ground truth.

Animals, campfires and obstacles are read from the world SDF, so the script
follows the world file instead of a hard-coded list. Existing animal models are
moved with the UserCommands set_pose service (nothing is spawned); campfires
stay where they are.

Every animal is placed
  - inside the survey radius around the Husky start (origin),
  - at least --min-separation from every other heat source (animals and
    campfires): mission_node treats records within 2.5 m as the same animal,
  - at least --obstacle-clearance from every tree / rock / tent centre, and
  - at least --min-start-distance from the Husky start.

Profiles
  nominal  animals in the open (default obstacle clearance 3.0 m)
  cover    each animal 1.0-2.0 m from a tree, so it is partly hidden from
           some directions (harder for thermal + YOLO)

The ground truth for the evaluator (same JSON format as
wildlife_spatial/config/static_large_demo_ground_truth.json, map frame = world
frame because the Husky starts at the origin) is written to --ground-truth-out.

Examples (simulation already running):
    ros2 run 41068_ignition_bringup random_animal_spawn.py --seed 1
    ros2 run 41068_ignition_bringup random_animal_spawn.py --seed 7 --profile cover
    ros2 run 41068_ignition_bringup random_animal_spawn.py --seed 1 --dry-run
Then start the mission with the printed ground_truth_file:=... argument.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree

# Model URI name -> evaluated class; must match the evaluator.
try:
    from wildlife_spatial.spatial_evaluator import WORLD_MODEL_CLASSES
except ImportError:  # wildlife_spatial not built: same table
    WORLD_MODEL_CLASSES = {
        'boar': 'boar', 'male_deer': 'deer', 'female_deer': 'deer',
        'kangaroo': 'kangaroo', 'campfire': 'campfire',
    }

ANIMAL_MODELS = {m for m, c in WORLD_MODEL_CLASSES.items() if c != 'campfire'}
GROUND_MODELS = {'forest_plane', 'grass_plane'}
MAX_ATTEMPTS = 20000


# --------------------------------------------------------------------------
# World parsing and layout (pure, tested offline)
# --------------------------------------------------------------------------

def load_world(path):
    """Return (world name, includes) of an SDF world.

    Each include is a dict with name, model (URI name), x, y, z, yaw.
    """
    world = ElementTree.parse(Path(path).expanduser()).getroot().find('world')
    if world is None:
        raise ValueError(f'{path} contains no <world> element')
    includes = []
    for include in world.findall('include'):
        uri = (include.findtext('uri') or '').strip()
        model = uri.split('://', 1)[-1].strip('/').split('/')[0]
        pose = [float(v) for v in (include.findtext('pose') or '0 0 0').split()]
        pose += [0.0] * (6 - len(pose))
        includes.append({
            'name': (include.findtext('name') or model).strip(),
            'model': model,
            'x': pose[0], 'y': pose[1], 'z': pose[2], 'yaw': pose[5],
        })
    return world.get('name', ''), includes


def split_world(includes):
    """Split includes into (animals, campfires, obstacles)."""
    animals = [i for i in includes if i['model'] in ANIMAL_MODELS]
    campfires = [i for i in includes
                 if WORLD_MODEL_CLASSES.get(i['model']) == 'campfire']
    obstacles = [i for i in includes
                 if i['model'] not in WORLD_MODEL_CLASSES
                 and i['model'] not in GROUND_MODELS]
    return animals, campfires, obstacles


def _min_distance(x, y, points):
    return min((math.hypot(x - p['x'], y - p['y']) for p in points),
               default=math.inf)


def choose_layout(rng, animals, campfires, obstacles, radius=22.0,
                  min_separation=6.0, obstacle_clearance=3.0,
                  min_start_distance=5.0, profile='nominal'):
    """Return {animal name: (x, y, yaw)} satisfying the placement rules."""
    trees = [o for o in obstacles if 'tree' in o['model']]
    placed = [dict(c) for c in campfires]
    layout = {}
    for animal in animals:
        for _ in range(MAX_ATTEMPTS):
            r = radius * math.sqrt(rng.random())
            a = rng.uniform(-math.pi, math.pi)
            # Round before checking so the stored position obeys the rules.
            x, y = round(r * math.cos(a), 2), round(r * math.sin(a), 2)
            if math.hypot(x, y) < min_start_distance:
                continue
            if _min_distance(x, y, placed) < min_separation:
                continue
            if profile == 'cover':
                if _min_distance(x, y, obstacles) < 1.0:
                    continue
                if not 1.0 <= _min_distance(x, y, trees) <= 2.0:
                    continue
            elif _min_distance(x, y, obstacles) < obstacle_clearance:
                continue
            yaw = rng.uniform(-math.pi, math.pi)
            layout[animal['name']] = (x, y, round(yaw, 3))
            placed.append({'x': x, 'y': y})
            break
        else:
            raise RuntimeError(
                f'No valid position for {animal["name"]} after {MAX_ATTEMPTS} '
                'attempts: relax --min-separation / --obstacle-clearance or '
                'increase --radius.')
    return layout


def ground_truth_payload(animals, campfires, layout, frame_id='husky1_map',
                         source=''):
    """Ground truth JSON (evaluator format) for the moved animals + campfires."""
    objects = []
    for animal in animals:
        x, y, _ = layout.get(animal['name'], (animal['x'], animal['y'], 0.0))
        objects.append({
            'id': animal['name'],
            'class_name': WORLD_MODEL_CLASSES[animal['model']],
            'position': {'x': x, 'y': y, 'z': animal['z']},
        })
    for fire in campfires:
        objects.append({
            'id': fire['name'],
            'class_name': 'campfire',
            'position': {'x': fire['x'], 'y': fire['y'], 'z': fire['z']},
        })
    return {'frame_id': frame_id, 'source': source, 'objects': objects}


# --------------------------------------------------------------------------
# Gazebo
# --------------------------------------------------------------------------

def detect_gazebo_cli():
    if shutil.which('ign'):
        return 'ign', 'ignition.msgs'
    if shutil.which('gz'):
        return 'gz', 'gz.msgs'
    raise RuntimeError('Neither `ign` nor `gz` was found in PATH.')


def wait_for_service(cli, service, timeout_seconds=20.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            result = subprocess.run([cli, 'service', '--list'],
                                    capture_output=True, text=True, timeout=3)
            if service in result.stdout:
                return True
        except (subprocess.SubprocessError, OSError):
            pass
        time.sleep(0.5)
    return False


def set_pose(cli, msg_ns, world, model_name, x, y, z, yaw):
    """Move one existing model with the UserCommands set_pose service."""
    request = (
        f'name: "{model_name}", '
        f'position: {{x: {x:.6f}, y: {y:.6f}, z: {z:.6f}}}, '
        f'orientation: {{x: 0.0, y: 0.0, z: {math.sin(yaw / 2):.9f}, '
        f'w: {math.cos(yaw / 2):.9f}}}'
    )
    cmd = [cli, 'service', '-s', f'/world/{world}/set_pose',
           '--reqtype', f'{msg_ns}.Pose', '--reptype', f'{msg_ns}.Boolean',
           '--timeout', '3000', '--req', request]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'Failed to move {model_name}: '
                           f'{(result.stdout + result.stderr).strip()}')


def default_world_path():
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('41068_ignition_bringup')) \
            / 'worlds' / 'large_demo.sdf'
    except Exception:  # not installed: source tree
        return Path(__file__).resolve().parents[1] / 'worlds' / 'large_demo.sdf'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--seed', type=int, default=None,
                        help='Fixed seed: the same layout every time.')
    parser.add_argument('--profile', choices=('nominal', 'cover'), default='nominal')
    parser.add_argument('--world', default=None,
                        help='World SDF (default: installed large_demo.sdf).')
    parser.add_argument('--radius', type=float, default=22.0,
                        help='Max distance of an animal from the start [m].')
    parser.add_argument('--min-separation', type=float, default=6.0)
    parser.add_argument('--obstacle-clearance', type=float, default=3.0)
    parser.add_argument('--min-start-distance', type=float, default=5.0)
    parser.add_argument('--ground-truth-out', default=None,
                        help='Default: ~/wildtrack_runs/ground_truth_<world>_'
                             '<profile>_seed<seed>.json')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the layout and write the ground truth, '
                             'but do not move Gazebo models.')
    args = parser.parse_args()

    world_path = Path(args.world).expanduser() if args.world else default_world_path()
    world_name, includes = load_world(world_path)
    animals, campfires, obstacles = split_world(includes)
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    layout = choose_layout(
        random.Random(seed), animals, campfires, obstacles, args.radius,
        args.min_separation, args.obstacle_clearance, args.min_start_distance,
        args.profile)

    print(f'World {world_name} ({world_path}), profile {args.profile}, seed {seed}')
    print(f'{"animal":16s} {"x (m)":>8s} {"y (m)":>8s} {"yaw (deg)":>10s}')
    for name, (x, y, yaw) in layout.items():
        print(f'{name:16s} {x:8.2f} {y:8.2f} {math.degrees(yaw):10.1f}')

    out = Path(args.ground_truth_out).expanduser() if args.ground_truth_out else (
        Path(os.path.expanduser('~/wildtrack_runs'))
        / f'ground_truth_{world_name}_{args.profile}_seed{seed}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = ground_truth_payload(
        animals, campfires, layout,
        source=f'random_animal_spawn.py --seed {seed} --profile {args.profile} '
               f'({world_path.name})')
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'\nGround truth: {out}')

    if args.dry_run:
        print('Dry run: Gazebo was not changed.')
        return 0

    try:
        cli, msg_ns = detect_gazebo_cli()
    except RuntimeError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    service = f'/world/{world_name}/set_pose'
    if not wait_for_service(cli, service):
        print(f'ERROR: {service} not found. Is the simulation running?',
              file=sys.stderr)
        return 1
    for animal in animals:
        x, y, yaw = layout[animal['name']]
        try:
            set_pose(cli, msg_ns, world_name, animal['name'], x, y, animal['z'], yaw)
        except RuntimeError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 1
    print(f'Moved {len(animals)} animals. Start the mission with:\n'
          f'  ros2 launch 41068_ignition_bringup 41068_mission.launch.py '
          f'ground_truth_file:={out} run_label:={args.profile}_seed{seed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
