import os

from custom_backends import MultipleResponse

from mock_data import MockDataset

# set the working directory to the folder containing this script
os.chdir(os.path.dirname(__file__))

# register the MultipleResponse backend
MockDataset.register_backend(MultipleResponse)

mock = MockDataset.read_yaml_spec("sbl.yaml")

mock_df = mock.generate_mock_data(nrows=100)

mock_df.to_csv("fake_data.csv", index=False)
