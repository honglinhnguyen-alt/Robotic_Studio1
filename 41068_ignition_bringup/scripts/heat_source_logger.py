#!/usr/bin/env python3
"""Heat source log: where each heat source is, what YOLO saw per trial, result.

mission_node publishes one line per event on /husky1/mission/log, in the order
it happened. This node prints them in colour, saves them to a file and prints a
table at the end. Run it in its own terminal (it only listens):

  ros2 run 41068_ignition_bringup heat_source_logger.py --ros-args -p use_sim_time:=true

Example:

  t=   0.5s NEW HEAT SOURCE candidate_1 at (9.30, -4.70)
  t=   2.3s candidate_1  Trial 1: Husky -> observation pose (6.60, -3.30), 3.0 m from it
  t=   9.8s candidate_1  Trial 1: YOLO -> no trained animal (unknown heat source, 0/5 frames)
  t=   9.8s candidate_1  Trial 2: Husky -> observation pose (7.30, -5.60), 2.2 m from it
  t=  17.3s candidate_1  Trial 2: YOLO -> no trained animal (unknown heat source, 0/5 frames)
  t=  17.3s RESULT candidate_1 at (9.30, -4.70): UNRESOLVED (no trained animal seen) after 2 trial(s)
  t=  17.3s candidate_2  Trial 1: Husky -> observation pose (15.30, -3.90), 3.0 m from it
  t=  24.8s candidate_2  Trial 1: YOLO -> boar 0.87 (exotic, 4/5 frames)
  t=  24.8s RESULT candidate_2 at (17.40, -1.70): BOAR (EXOTIC), confidence 0.87, after 1 trial(s)

Lines are also saved (without colours) to
~/wildtrack_runs/heat_sources_<date>_<time>.log.

Run record (E3): the node also listens to the mission status, the coverage
tracker and the spatial evaluator. When the mission reaches COMPLETE (or the
node is stopped first) it writes

  ~/wildtrack_runs/mission_summary_<date>_<time>.json   full record
  ~/wildtrack_runs/runs.csv                              one row per run

Duration is ROS time (simulation time with use_sim_time) from the first
status after INITIALISING until COMPLETE. Set run_label / world to tell the
rows of different scenarios apart.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String

YELLOW, GREEN, CYAN, RED, BOLD, RESET = (
    '\033[33m', '\033[32m', '\033[36m', '\033[31m', '\033[1m', '\033[0m')

# One row per run in runs.csv (E3). Keep the order stable: rows from earlier
# runs are not rewritten.
RUN_CSV_FIELDS = [
    'stamp', 'run_label', 'world', 'completed', 'duration_s', 'wall_duration_s',
    'exploration_status', 'candidates', 'exotic', 'native',
    'unresolved', 'unresolved_low_confidence', 'duplicate', 'pending',
    'lidar_coverage_pct', 'thermal_coverage_pct',
    'truth_objects', 'matched', 'false_positives', 'precision', 'recall',
    'mean_planar_error_m', 'max_planar_error_m', 'within_tolerance',
    'classification_accuracy', 'boars', 'boars_detected', 'boars_classified',
    'log_file',
]


def _pct(value):
    return '' if value is None else round(100.0 * float(value), 1)


def _round(value, digits=3):
    return '' if value is None else round(float(value), digits)


def upsert_run_row(csv_path, row):
    """Replace this run's row in runs.csv (same stamp) or append it."""
    rows = []
    if csv_path.exists():
        with csv_path.open(newline='') as csv_file:
            rows = [r for r in csv.DictReader(csv_file) if r.get('stamp') != str(row['stamp'])]
    rows.append({key: row.get(key, '') for key in RUN_CSV_FIELDS})
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RUN_CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def run_record(stamp, run_label, world, completed, mission_start, mission_end,
               wall_seconds, status, coverage, evaluation, log_file):
    """Build the JSON run record and its runs.csv row (pure, for tests)."""
    status = status or {}
    coverage = coverage or {}
    evaluation = evaluation or {}
    counts = status.get('counts') or {}
    boar = (evaluation.get('per_class') or {}).get('boar') or {}
    duration = (None if mission_start is None or mission_end is None
                else max(0.0, mission_end - mission_start))
    row = {
        'stamp': stamp,
        'run_label': run_label,
        'world': world,
        'completed': completed,
        'duration_s': _round(duration, 1),
        'wall_duration_s': _round(wall_seconds, 1),
        'exploration_status': status.get('exploration_status', ''),
        'candidates': len(status.get('candidates') or []),
        'lidar_coverage_pct': _pct(coverage.get('lidar_coverage')),
        'thermal_coverage_pct': _pct(coverage.get('thermal_coverage')),
        'truth_objects': evaluation.get('ground_truth_count', ''),
        'matched': evaluation.get('matched_count', ''),
        'false_positives': evaluation.get('false_positive_count', ''),
        'precision': _round(evaluation.get('precision')),
        'recall': _round(evaluation.get('recall')),
        'mean_planar_error_m': _round(evaluation.get('mean_planar_error_m')),
        'max_planar_error_m': _round(evaluation.get('maximum_planar_error_m')),
        'within_tolerance': evaluation.get('within_tolerance_count', ''),
        'classification_accuracy': _round(evaluation.get('classification_accuracy')),
        'boars': boar.get('ground_truth', ''),
        'boars_detected': boar.get('detected', ''),
        'boars_classified': boar.get('correctly_classified', ''),
        'log_file': log_file,
    }
    for key in ('exotic', 'native', 'unresolved', 'unresolved_low_confidence',
                'duplicate', 'pending'):
        row[key] = counts.get(key, '')
    return {
        'row': row,
        'final_mission_status': status,
        'coverage': coverage,
        'evaluation': evaluation,
    }


class HeatSourceLogger(Node):

    def __init__(self):
        super().__init__('heat_source_logger')
        self.declare_parameter('mission_log_topic', '/husky1/mission/log')
        self.declare_parameter('log_dir', '~/wildtrack_runs')
        self.declare_parameter('color', True)
        self.declare_parameter('mission_status_topic', '/husky1/mission/status')
        self.declare_parameter('coverage_topic', '/husky1/mission/coverage')
        self.declare_parameter('evaluation_topic', '/husky1/evaluation/summary')
        self.declare_parameter('run_label', '')
        self.declare_parameter('world', '')
        self.color = bool(self.get_parameter('color').value)
        self.run_label = str(self.get_parameter('run_label').value)
        self.world = str(self.get_parameter('world').value)

        self.log_dir = Path(os.path.expanduser(str(self.get_parameter('log_dir').value)))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = self.log_dir / f'heat_sources_{self.run_stamp}.log'
        self.log_file = open(self.log_path, 'w')

        self.t0 = None
        self.sources = {}   # candidate -> {'where', 'trials', 'result'}
        self.create_subscription(
            String, str(self.get_parameter('mission_log_topic').value), self._event, 50)

        # Run record (E3)
        self.status = None
        self.coverage = None
        self.evaluation = None
        self.mission_start = None       # ROS time [s] of the first active status
        self.mission_end = None
        self.wall_start = datetime.now()
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            String, str(self.get_parameter('mission_status_topic').value), self._status, 10)
        self.create_subscription(
            String, str(self.get_parameter('coverage_topic').value), self._coverage, 10)
        self.create_subscription(
            String, str(self.get_parameter('evaluation_topic').value), self._evaluation,
            latched)
        self._out(f'Heat source log -> {self.log_path}', BOLD)

    def _out(self, text, colour=''):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None and now > 0.0:
            self.t0 = now
        line = f't={now - (self.t0 or now):7.1f}s {text}'
        print((colour + line + RESET) if (self.color and colour) else line, flush=True)
        self.log_file.write(line + '\n')
        self.log_file.flush()

    def _event(self, msg):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        kind = event.get('kind', '')
        candidate = event.get('candidate_id', '')
        text = event.get('text', '')
        if kind == 'sweep':      # thermal coverage sweep, not a heat source
            self._out(text, CYAN)
            return
        source = self.sources.setdefault(candidate, {'where': '-', 'trials': 0,
                                                     'result': 'pending'})
        if event.get('x') is not None:
            source['where'] = f'({float(event["x"]):.2f}, {float(event["y"]):.2f})'

        if kind == 'new':
            self._out(text, YELLOW)
        elif kind == 'result':
            source['result'] = event.get('summary', text)
            source['trials'] = event.get('trials', source['trials'])
            state = event.get('state', '')
            colour = GREEN if state.startswith('confirmed') else (
                RED if state == 'unresolved' else YELLOW)
            self._out(text, colour)
        else:   # going / trial
            if text.startswith('Trial '):
                try:
                    source['trials'] = max(source['trials'], int(text.split()[1].rstrip(':')))
                except (IndexError, ValueError):
                    pass
            self._out(f'{candidate:<12} {text}', CYAN if kind == 'going' else '')

    @staticmethod
    def _parse(msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def _coverage(self, msg):
        self.coverage = self._parse(msg) or self.coverage

    def _evaluation(self, msg):
        self.evaluation = self._parse(msg) or self.evaluation

    def _status(self, msg):
        status = self._parse(msg)
        if status is None:
            return
        previous = (self.status or {}).get('mission_state')
        self.status = status
        state = status.get('mission_state')
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.mission_start is None and state not in (None, 'INITIALISING'):
            self.mission_start = now
        # mission_node leaves COMPLETE again if a new candidate appears, so
        # the record is rewritten on every entry into COMPLETE.
        if state == 'COMPLETE' and previous != 'COMPLETE':
            self.mission_end = now
            self._out('MISSION COMPLETE', GREEN)
            self.write_run_record()

    def write_run_record(self):
        """Write mission_summary_<stamp>.json and this run's row in runs.csv."""
        completed = (self.status or {}).get('mission_state') == 'COMPLETE'
        record = run_record(
            stamp=self.run_stamp, run_label=self.run_label, world=self.world,
            completed=completed, mission_start=self.mission_start,
            mission_end=self.mission_end if completed else
            self.get_clock().now().nanoseconds * 1e-9,
            wall_seconds=(datetime.now() - self.wall_start).total_seconds(),
            status=self.status, coverage=self.coverage, evaluation=self.evaluation,
            log_file=str(self.log_path))
        summary_path = self.log_dir / f'mission_summary_{self.run_stamp}.json'
        summary_path.write_text(json.dumps(record, indent=2) + '\n')
        csv_path = self.log_dir / 'runs.csv'
        upsert_run_row(csv_path, record['row'])
        self._out(f'run record: {summary_path} (row {self.run_stamp} in {csv_path})', BOLD)

    def summary(self):
        self._out('=' * 78, BOLD)
        self._out(f'{"heat source":<13}{"location":<18}{"trials":<8}result', BOLD)
        for candidate, s in self.sources.items():
            self._out(f'{candidate:<13}{s["where"]:<18}{s["trials"]:<8}{s["result"]}')
        self._out(f'saved: {self.log_path}', BOLD)
        self.log_file.close()


def main(args=None):
    rclpy.init(args=args)
    node = HeatSourceLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stopped before COMPLETE: still keep the run, marked incomplete.
        node.write_run_record()
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
