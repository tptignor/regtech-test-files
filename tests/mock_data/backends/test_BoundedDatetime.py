import pytest

from mock_data.backends import BoundedDatetime


def test_lower_and_upper_bounds_are_correctly_set():
    bounded_dt = BoundedDatetime(min_datetime="20200101", max_datetime="20211222")
    # these are the unix timestamps associated with the above dates
    assert bounded_dt._lower_bound == 1577854800.0
    assert bounded_dt._upper_bound == 1640149200.0


def test_lower_bound_after_upper_bound_fails():
    with pytest.raises(ValueError):
        BoundedDatetime(min_datetime="20220101", max_datetime="20210101")


def test_invalid_date_format_fails():
    with pytest.raises(ValueError):
        BoundedDatetime(min_datetime="20220101", max_datetime="20210101", format="%bad")


def test_alternative_formatting_is_valid():
    bounded_dt = BoundedDatetime(
        min_datetime="December 01, 2010",
        max_datetime="December 31, 2012",
        format="%B %d, %Y",
    )

    assert bounded_dt._lower_bound == 1291179600.0
    assert bounded_dt._upper_bound == 1356930000.0


def test_all_samples_are_within_expected_set_of_values():
    bounded_dt = BoundedDatetime(
        min_datetime="December 20, 2020",
        max_datetime="December 25, 2020",
        format="%B %d, %Y",
    )

    samples = bounded_dt.generate_samples(size=500)

    for sample in samples:
        assert sample in [f"December {d}, 2020" for d in range(20, 26)]
