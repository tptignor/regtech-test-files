"""These are tests related to the BoundedNumerical backend class."""

import numpy as np
import pytest
from scipy.stats import alpha, norm, pearsonr

from mock_data.backends import BoundedNumerical


def test_valid_distributions_are_accepted_with_no_arguments():
    # these distributions do not require any keyword parameters
    for distribution in ("arcsine", "norm", "laplace"):
        try:
            BoundedNumerical(distribution=distribution)
        except (ValueError, TypeError) as errors:
            assert False, "Invalid distribution supplied"


def test_distribution_kwargs_are_fed_to_distribution_object():
    alpha = BoundedNumerical(distribution="alpha", a=1)
    assert alpha.distribution.kwds["a"] == 1

    chi2 = BoundedNumerical(distribution="chi2", df=33)
    assert chi2.distribution.kwds["df"] == 33


def test_instance_creation_with_default_parameters():
    try:
        BoundedNumerical()
    except Exception:
        assert False, "Something didn't work"


def test_ensure_samples_are_all_within_specified_upper_and_lower_bounds():
    # tests a normal distribution and chi squared distribution with arbitrary upper
    # and lower bounds and ensures all samples fall within said bounds

    norm_sampler = BoundedNumerical(distribution="norm", lower_bound=3, upper_bound=22)
    norm_samples = norm_sampler.generate_samples(size=10000)
    assert (norm_samples >= 3).all() and (norm_samples <= 22).all()

    chi2_sampler = BoundedNumerical(
        distribution="chi2", lower_bound=-222, upper_bound=3, df=5
    )
    chi2_samples = chi2_sampler.generate_samples(size=10000)
    assert (chi2_samples >= -222).all() and (chi2_samples <= 3).all()


def test_upper_bound_less_than_lower_bound_raises_ValueError():
    with pytest.raises(ValueError):
        BoundedNumerical(lower_bound=1, upper_bound=0)


def test_upper_bound_equal_to_lower_bound_raises_ValueError():
    with pytest.raises(ValueError):
        BoundedNumerical(lower_bound=1, upper_bound=1)


def test_coerce_to_int_flag_produces_only_integer_samples_within_specified_bounds():
    # going to arbitrarily use an f distribution here with dfn=13, dfd=41
    f_sampler = BoundedNumerical(
        distribution="f",
        lower_bound=0,
        upper_bound=100,
        coerce_to_int=True,
        dfn=13,
        dfd=41,
    )

    f_samples = f_sampler.generate_samples(size=10000)

    assert (f_samples.astype(int) == f_samples).all()
    assert (f_samples >= 0).all() and (f_samples <= 100).all()


def test_generated_samples_conform_to_specified_distribution():
    """This test verifies that scaled and shifted samples still comform
    to whichever underlying distribution has been specified. By testing
    the correlation of two numerical CDFs you can determine whether the
    distributions align in a scale and location agnostic manner. This is
    basically the Shapiro-Wilk test.

    Here we perform validation on a normal distribution and an alpha
    distribution with a=4 set. We test at 99% confidence with a maximum
    P-value of 0.01."""

    # let's start with a normal distribution
    scaled_norm_sampler = BoundedNumerical(distribution="norm")
    sorted_and_scaled_norm_samples = np.sort(
        scaled_norm_sampler.generate_samples(size=10000)
    )

    scipy_norm_samples = np.sort(norm.rvs(size=10000))

    p_stat_norm = pearsonr(sorted_and_scaled_norm_samples, scipy_norm_samples)
    assert p_stat_norm.pvalue <= 0.01

    # Now let's try an alpha distribution
    scaled_alpha_sampler = BoundedNumerical(distribution="alpha", a=4)
    sorted_and_scaled_alpha_samples = np.sort(
        scaled_alpha_sampler.generate_samples(size=10000)
    )

    scipy_alpha_samples = np.sort(alpha.rvs(a=4, size=10000))

    p_stat_alpha = pearsonr(sorted_and_scaled_alpha_samples, scipy_alpha_samples)
    assert p_stat_alpha.pvalue <= 0.01
