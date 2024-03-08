"""Illustrates using the mock_data package to generate a mock dataset of 100 rows based
on the yaml spec example_spec.yaml. The mock dataframe is saved to a csv called 
example_mock_data.csv."""

try:
    from mock_data import MockDataset
except ModuleNotFoundError:
    raise RuntimeError("You must first pip install mock_data via `pip install .`")

# instance of the mock dataset class
mock = MockDataset.read_yaml_spec("example_spec.yaml")

# a Pandas dataframe containing 100 rows
df = mock.generate_mock_data(nrows=100)

# write to csv
df.to_csv("example_mock_data.csv", index=False, float_format="%.2f")
