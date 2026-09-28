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
  t=  17.3s RESULT candidate_1 at (9.30, -4.70): NOT A TRAINED ANIMAL (heat source rejected) ...
  t=  17.3s candidate_2  Trial 1: Husky -> observation pose (15.30, -3.90), 3.0 m from it
  t=  24.8s candidate_2  Trial 1: YOLO -> boar 0.87 (exotic, 4/5 frames)
  t=  24.8s RESULT candidate_2 at (17.40, -1.70): BOAR (EXOTIC), confidence 0.87, after 1 trial(s)

Lines are also saved (without colours) to
~/wildtrack_runs/heat_sources_<date>_<time>.log.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

YELLOW, GREEN, CYAN, RED, BOLD, RESET = (
    '\033[33m', '\033[32m', '\033[36m', '\033[31m', '\033[1m', '\033[0m')


class HeatSourceLogger(Node):

    def __init__(self):
        super().__init__('heat_source_logger')
        self.declare_parameter('mission_log_topic', '/husky1/mission/log')
        self.declare_parameter('log_dir', '~/wildtrack_runs')
        self.declare_parameter('color', True)
        self.color = bool(self.get_parameter('color').value)

        log_dir = Path(os.path.expanduser(str(self.get_parameter('log_dir').value)))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / datetime.now().strftime('heat_sources_%Y%m%d_%H%M%S.log')
        self.log_file = open(self.log_path, 'w')

        self.t0 = None
        self.sources = {}   # candidate -> {'where', 'trials', 'result'}
        self.create_subscription(
            String, str(self.get_parameter('mission_log_topic').value), self._event, 50)
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
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()