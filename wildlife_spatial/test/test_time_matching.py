from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2

from wildlife_spatial.candidate_localizer import closest_cloud
from wildlife_spatial.candidate_localizer import stamp_nanoseconds


def make_cloud(seconds, nanoseconds=0):
    cloud = PointCloud2()
    cloud.header.stamp = Time(sec=seconds, nanosec=nanoseconds)
    return cloud


def test_stamp_nanoseconds():
    stamp = Time(sec=12, nanosec=345)

    assert stamp_nanoseconds(stamp) == 12_000_000_345


def test_closest_cloud_returns_empty_result_for_empty_buffer():
    cloud, difference = closest_cloud([], Time(sec=10))

    assert cloud is None
    assert difference is None


def test_closest_cloud_selects_nearest_timestamp():
    clouds = [
        make_cloud(9, 800_000_000),
        make_cloud(10, 40_000_000),
        make_cloud(10, 300_000_000),
    ]

    cloud, difference = closest_cloud(
        clouds,
        Time(sec=10, nanosec=0),
    )

    assert cloud is clouds[1]
    assert abs(difference - 0.04) < 1e-9
