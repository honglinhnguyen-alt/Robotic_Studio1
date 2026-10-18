import json

import pytest

from wildlife_gui.mission_panel import describe_activity
from wildlife_gui.mission_panel import evaluation_line
from wildlife_gui.mission_panel import parse_mission_status
from wildlife_gui.mission_panel import summary_lines


def candidate(identifier, state, class_name, confidence, x, y,
              attempts=1, last_outcome=None, reason='', reason_detail=''):
    # Same keys as mission_node._publish_ui_status.
    return {
        'candidate_id': identifier,
        'position': {'x': x, 'y': y, 'z': 0.5},
        'state': state,
        'class_name': class_name,
        'confidence': confidence,
        'observation_count': 4,
        'attempts': attempts,
        'last_outcome': last_outcome,
        'reason': reason,
        'reason_detail': reason_detail,
    }


def status_message(mission_state='INVESTIGATING', candidates=None, **extra):
    payload = {
        'frame_id': 'husky1_map',
        'mission_state': mission_state,
        'exploration_status': 'exploring',
        'exploration_paused': True,
        'robot': {'x': 1.0, 'y': 2.0, 'yaw': 0.1},
        'current_candidate': 'candidate_2',
        'attempt': 2,
        'phase': 'waiting_husky',
        'husky_phase': 'navigating',
        'counts': {'exotic': 1, 'native': 0, 'unresolved_low_confidence': 0,
                   'unresolved': 1, 'pending': 1, 'duplicate': 0},
        'candidate_count': 0,
        'candidates': candidates or [],
    }
    payload.update(extra)
    return json.dumps(payload)


CANDIDATES = [
    candidate('candidate_1', 'confirmed_exotic', 'boar', 0.91, 1.2, -12.1,
              last_outcome='exotic'),
    candidate('candidate_2', 'investigating', 'unknown_heat_source', 0.0,
              4.0, 10.0, attempts=2, last_outcome='too_far'),
    candidate('candidate_3', 'unresolved', 'unresolved', None,
              9.0, -5.0, attempts=2, last_outcome='no_animal',
              reason='no_trained_animal_seen'),
    candidate('candidate_4', 'unresolved', 'unknown_heat_source', 0.3,
              -13.0, 6.0, attempts=3, last_outcome='unreachable',
              reason='unreachable'),
]


def test_parse_mission_status_reads_mission_node_payload():
    status = parse_mission_status(status_message(candidates=CANDIDATES))

    assert status['mission_state'] == 'INVESTIGATING'
    assert status['counts']['exotic'] == 1
    assert len(status['candidates']) == 4
    assert status['candidates'][0]['state'] == 'confirmed_exotic'
    assert status['candidates'][2]['confidence'] == pytest.approx(0.0)
    assert status['candidates'][1]['last_outcome'] == 'too_far'


def test_parse_mission_status_rejects_bad_payload():
    with pytest.raises(ValueError):
        parse_mission_status(json.dumps({'counts': {}, 'candidates': []}))
    with pytest.raises(ValueError):
        parse_mission_status(status_message(candidates=[
            candidate('candidate_1', 'unresolved', 'x', float('nan'), 0, 0)
        ]))


def test_activity_names_current_candidate_and_attempt():
    status = parse_mission_status(status_message())
    assert describe_activity(status) == (
        'Candidate candidate_2, attempt 2 (navigating)'
    )

    status = parse_mission_status(status_message(
        mission_state='COVERAGE', exploration_paused=False,
        current_candidate=None))
    assert 'Searching for heat sources' in describe_activity(status)


def test_summary_lists_exotic_and_unresolved_candidates():
    lines = summary_lines(parse_mission_status(
        status_message(mission_state='COMPLETE', candidates=CANDIDATES)))

    assert lines[0].startswith('Exotic: 1')
    assert 'Exotic animals:' in lines
    assert any('candidate_1: boar 91% at (1.2, -12.1) m' in line
               for line in lines)
    assert 'Needs a manual check (possible animals first):' in lines
    assert any('candidate_4' in line and 'no observation pose reachable' in line
               for line in lines)
    assert 'Native animals:' not in lines


def test_evaluation_line_reports_boar_recall():
    text = json.dumps({
        'matched_count': 3,
        'ground_truth_count': 14,
        'mean_planar_error_m': 0.324,
        'localization_tolerance_m': 1.0,
        'within_tolerance_count': 3,
        'per_class': {'boar': {'correctly_classified': 1,
                               'ground_truth': 5}},
    })

    assert evaluation_line(text) == (
        'Matched 3/14 heat sources | mean error 0.32 m | '
        'within 1.0 m: 3 | boars correctly classified 1/5'
    )


def test_manual_check_lists_possible_animals_first():
    rows = [
        candidate('candidate_7', 'unresolved', 'unresolved', 0.0, 1, 1,
                  reason='unreachable'),
        candidate('candidate_8', 'unresolved', 'unresolved', 0.0, 2, 2,
                  reason='no_trained_animal_seen'),
        candidate('candidate_9', 'unresolved', 'unresolved', 0.66, 3, 3,
                  reason='low_confidence', reason_detail='boar 0.66'),
    ]
    lines = summary_lines(parse_mission_status(
        status_message(mission_state='COMPLETE', candidates=rows)))
    start = lines.index('Needs a manual check (possible animals first):')
    assert 'candidate_9' in lines[start + 1]
    assert 'possible animal, low confidence: boar 0.66' in lines[start + 1]
    assert 'candidate_8' in lines[start + 2]
    assert 'candidate_7' in lines[start + 3]
