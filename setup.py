from setuptools import setup, find_packages

VERSION = '0.1.0' 
DESCRIPTION = 'Wavefront sensing and control algorithms and tools being developed in UASAL'
LONG_DESCRIPTION = 'Wavefront sensing and control algorithms and tools being developed in UASAL'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="lina",
        version=VERSION,
        author="Kevin Derby",
        author_email="<derbyk@arizona.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'lina', 'wfsc', 'wfs&c', 'wavefront', 'sensing', 'control'],
        classifiers= [
            "Development Status :: Alpha-0.1.0",
            "Programming Language :: Python :: 3",
        ]
)
