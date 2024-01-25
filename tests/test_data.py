import pytest
import sccoral


@pytest.fixture(scope="module")
def tmp_dataset_dir(tmp_path_factory):
    """See scanpy testing"""
    new_dir = tmp_path_factory.mktemp("tmp_data")
    yield new_dir


@pytest.mark.network
def test_splatter_simulation(tmp_dataset_dir):
    adata = sccoral.data.splatter_simulation(tmp_dataset_dir, "simulation.h5ad")
    assert adata.shape == (4000, 5000)
