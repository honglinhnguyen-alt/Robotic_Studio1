import numpy as np

from wildlife_spatial.thermal_contrast import hot_mask, local_background_k


def scene():
    k = np.full((120, 160), 293.0)        # ground
    k[:50, :] = 300.7                     # trees / rocks
    k[60:80, 20:40] = 330.0               # normal animal
    k[60:80, 100:120] = 312.0             # low-contrast animal
    return k


def test_relative_finds_both_animals_but_no_trees():
    hot = hot_mask(scene(), 'relative', 315.0, 10.0)
    assert hot[70, 30] and hot[70, 110]
    assert not hot[10, 10] and not hot[100, 10]


def test_absolute_315_misses_the_low_contrast_animal():
    hot = hot_mask(scene(), 'absolute', 315.0, 10.0)
    assert hot[70, 30] and not hot[70, 110]


def test_local_background_is_the_ring_around_the_box():
    k = scene()
    hot = hot_mask(k, 'relative', 315.0, 10.0)
    assert local_background_k(k, hot, (100, 60, 20, 20), 6) == 293.0


def test_cold_sky_does_not_pull_the_threshold_onto_the_trees():
    k = np.full((120, 160), 290.8)        # cold sky fills most of the frame
    k[40:70, :] = 300.7                   # band of trees
    k[70:, :] = 293.0                     # ground
    k[50:70, 70:90] = 329.0               # animal in front of the trees
    hot = hot_mask(k, 'relative', 315.0, 10.0)
    assert hot[60, 80]
    assert not hot[45, 10]                # trees stay cold
