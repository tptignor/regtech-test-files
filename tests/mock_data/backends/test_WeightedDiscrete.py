import pytest
from scipy.stats import binomtest

from mock_data.backends import WeightedDiscrete


def test_assignment_of_populations_and_weights():
    weighted_discrete = WeightedDiscrete(
        population={"Thing_A": 1, "Thing_B": 2, "Last_Thing": 3}
    )

    assert weighted_discrete._population == ["Thing_A", "Thing_B", "Last_Thing"]
    assert weighted_discrete._weights == [1, 2, 3]


def test_passing_a_list_to_constructor_is_properly_converted_to_frequency_dict():
    weighted_discrete = WeightedDiscrete(population=["A", "B", "C"])

    assert weighted_discrete._population == ["A", "B", "C"]
    assert weighted_discrete._weights == [1, 1, 1]


def test_assignment_of_populations_and_decimal_weights():
    weighted_discrete = WeightedDiscrete(
        population={"Thing_A": 0.1, "Thing_B": 0.2, "Last_Thing": 0.3}
    )

    assert weighted_discrete._population == ["Thing_A", "Thing_B", "Last_Thing"]
    assert weighted_discrete._weights == [0.1, 0.2, 0.3]


def test_negative_weights_are_not_accepted():
    with pytest.raises(ValueError):
        weighted_discrete = WeightedDiscrete(
            population={"Bad": -1, "Thing_B": 2, "Last_Thing": 3}
        )


def test_non_numerical_weights_are_not_accepted():
    with pytest.raises(TypeError):
        weighted_discrete = WeightedDiscrete(
            population={"Bad": "yolo", "Thing_B": 2, "Last_Thing": 3}
        )


def test_generated_samples_are_keys_of_frequency_distribution():
    weighted_discrete = WeightedDiscrete(population={"A": 1, "B": 1, "C": 1})
    samples = weighted_discrete.generate_samples(size=1000)
    assert all([s in ("A", "B", "C") for s in samples])


@pytest.mark.parametrize(
    "frequency_dist",
    [
        pytest.param({"A": 1, "B": 1, "C": 1}),
        pytest.param({"A": 0.1, "B": 0.1, "C": 0.1}),
        pytest.param({"A": 1}),
        pytest.param({"A": 1, "B": 2, "C": 6}),
    ],
)
def test_sampled_frequency_aligns_with_weights(frequency_dist):
    # this test runs the binomial distribution test on each element of
    # supplied distribution.

    n = 1000  # number of samples to generate. Fed to binomial test
    weighted_discrete = WeightedDiscrete(population=frequency_dist)
    samples = weighted_discrete.generate_samples(n)

    for key, weight in frequency_dist.items():
        k = sum([s == key for s in samples])
        p = weight / sum(weighted_discrete._weights)

        # null hypothesis is that the k samples were drawn from a sample
        # of size n with probability of occuring equal to p.
        assert binomtest(k, n, p).pvalue >= 0.01
