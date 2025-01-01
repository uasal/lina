# this convenience function returns an object that describes the observatory with units

import numpy as np
import astropy.units as u
import tomlkit
import pathlib
from astropy.table import Table
from scipy.interpolate import PchipInterpolator

class Config(object):
    """:ref:
        Observatory configuration class using astropy units
    Args:


    Attributes:

    """

    def __init__(self, data_path=None, title=None, **specs):
        self.data_path = data_path
        self.title = title
        for key, value in specs.items():
            if isinstance(value, dict):
                value = Config(data_path=data_path, title=f'{self.title}: {key}', **value)
            elif isinstance(value, str):
                if value:
                    data_file = data_path / pathlib.Path(value)
                    if data_file.exists():
                        table = Table.read(data_file)
                        value = PchipInterpolator(table.columns[0], table.columns[1])
                    else:
                        try:
                            value = u.Quantity(value)
                        except Exception:
                            print(Exception)
                            pass
                else:
                    value = None
            setattr(self, key, value)

    def __repr__(self):
        return f'<{self.title}>'

    @classmethod
    def read(cls, fname='params/*.toml', path=""):
        """
        Read observatory configuration from TOML file.

        Args:
            cls (type): The class object that this method belongs to.
            fname (str, optional): The filename pattern to search for TOML files.
                     Defaults to 'params/*.toml'.
            path (str, optional): The path to the directory containing the TOML files.
                  If not provided, the parent directory of the current script is used.

        Returns:    
            Config: An object containing the observatory parameters read from the TOML files.
        """

        if path is None:
            data_path = pathlib.Path(__file__).resolve().parent.parent
        else:
            data_path = pathlib.Path(path)
        config = {}
        for param_file in data_path.glob(fname):
            with open(param_file) as f:
                config.update(tomlkit.load(f))
        return Config(data_path=data_path, **config)


scope = Config.read()
