from importlib.metadata import version

from . import data, model, module, tl

package_name = "scvi-tools"
__version__ = version(package_name)

__all__ = ["model", "module", "data", "tl"]
