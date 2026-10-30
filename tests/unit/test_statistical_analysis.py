"""Statistical primitives and strict CSV pairing, without benchmark execution."""
import numpy as np
import pandas as pd
import pytest

from repath_evaluation import statistical_analysis as analysis


@pytest.fixture
def records():
    keys = [(site, str(i)) for site in sorted(analysis.SITES) for i in range(53)]
    return pd.DataFrame([
        {"algorithm": name, "site": site, "repeat": 1, "old_element_id": old,
         "expected_new_element_id": old, "correct": (i + j) % 3 == 0}
        for j, name in enumerate(analysis.ALGORITHMS)
        for i, (site, old) in enumerate(keys)
    ])


def write_load(tmp_path, records):
    path = tmp_path / "queries.csv"
    records.to_csv(path, index=False)
    return analysis.load_outcomes(path)


def test_pairing_is_by_query_and_only_repeat_one(tmp_path, records):
    expected = write_load(tmp_path, records)
    repeated = records.assign(repeat=2, correct=False)
    shuffled = pd.concat([records, repeated]).sample(frac=1, random_state=7)
    pd.testing.assert_frame_equal(write_load(tmp_path, shuffled), expected)
    assert expected.shape == (318, 6)
    assert not expected.REPATH.equals(expected.WATER)


@pytest.mark.parametrize('defect', [
    'duplicate', 'missing', 'key', 'target', 'correct', 'algorithm', 'site', 'empty',
])
def test_invalid_records_rejected(tmp_path, records, defect):
    if defect == 'duplicate':
        records = pd.concat([records, records.iloc[[0]]])
    elif defect == 'missing':
        records = records.iloc[1:]
    elif defect == 'key':
        records.loc[318, 'old_element_id'] = 'unmatched'
    elif defect == 'target':
        records.loc[318, 'expected_new_element_id'] = 'wrong'
    elif defect == 'correct':
        records['correct'] = records.correct.astype(object)
        records.loc[0, 'correct'] = 'unknown'
    elif defect == 'algorithm':
        records.loc[0, 'algorithm'] = 'unexpected'
    elif defect == 'site':
        records['site'] = records.site.replace({'yahoo': 'development-site'})
    elif defect == 'empty':
        records.loc[0, 'old_element_id'] = ''
    with pytest.raises(ValueError):
        write_load(tmp_path, records)


def test_contingency_and_exact_mcnemar():
    repath = np.array([True] * 302 + [False] * 16)
    baseline = np.array([True] * 255 + [False] * 47 + [True] * 4 + [False] * 12)
    assert analysis.contingency(repath, baseline) == (255, 47, 4, 12)
    outcomes = pd.DataFrame({name: baseline for name in analysis.ALGORITHMS})
    outcomes['REPATH'] = repath
    individual, comparisons = analysis.analyze(outcomes, 100, 42)
    row = comparisons.loc[comparisons.baseline == 'SFTM'].iloc[0]
    assert row.repath_correct == 302
    assert row.baseline_correct == 259
    assert row.mcnemar_p_value == pytest.approx(2.416307154e-10, rel=1e-9)
    assert row.accuracy_improvement_pp == pytest.approx(100 * 43 / 318)
    assert len(individual) == 6
    assert len(comparisons) == 5
    _, identical = analysis.analyze(outcomes.assign(REPATH=baseline), 10, 42)
    assert (identical.mcnemar_p_value == 1).all()
    assert (identical[['ci_lower_pp', 'ci_upper_pp']] == 0).all().all()


def test_holm_original_order_monotonicity_ties_and_cap():
    values = np.array([0.04, 0.001, 0.03, 0.01, 0.8])
    np.testing.assert_allclose(analysis.holm_adjust(values), [0.09, 0.005, 0.09, 0.04, 0.8])
    np.testing.assert_allclose(analysis.holm_adjust(np.array([0.6, 0.6])), [1, 1])


def test_bootstrap_uses_reproducible_shared_indices():
    values = np.array([[True, False], [False, True], [True, False]])
    count = 300  # Crosses the implementation's batch boundary.
    indices = np.random.default_rng(42).integers(0, 3, size=(count, 3))
    expected = values[indices].mean(axis=1) * 100
    actual = analysis.bootstrap_accuracies(values, count, 42)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, analysis.bootstrap_accuracies(values, count, 42))
    np.testing.assert_allclose(actual.sum(axis=1), 100)
    assert not np.array_equal(actual, analysis.bootstrap_accuracies(values, count, 43))
    with pytest.raises(ValueError):
        analysis.bootstrap_accuracies(values, 0, 42)


def test_report_intervals_match_shared_percentile_samples(tmp_path, records):
    outcomes = write_load(tmp_path, records)
    individual, paired = analysis.analyze(outcomes, 300, 42)
    samples = analysis.bootstrap_accuracies(outcomes.to_numpy(), 300, 42)
    np.testing.assert_allclose(individual.ci_lower_pct, np.quantile(samples, 0.025, axis=0))
    np.testing.assert_allclose(individual.ci_upper_pct, np.quantile(samples, 0.975, axis=0))
    differences = samples[:, [0]] - samples[:, 1:]
    np.testing.assert_allclose(paired.ci_lower_pp, np.quantile(differences, 0.025, axis=0))
    np.testing.assert_allclose(paired.ci_upper_pp, np.quantile(differences, 0.975, axis=0))
