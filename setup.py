import os
import fnmatch
from pathlib import Path

import pkg_resources

from grpc_tools import protoc
from setuptools import setup, find_packages


def find_files(pattern, root):
    """Return all the files matching pattern below root dir."""
    for dirpath, _, files in os.walk(root):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(dirpath, filename)


_VERSION = '0.0.1'
with open("requirements.txt") as requirements_file:
    REQUIRED_PACKAGES = requirements_file.read() \
                            .split('\n\n')[0] \
                            .split('\n')[1:]

_proto_include = pkg_resources.resource_filename('grpc_tools', '_proto')
_protobufs_path = \
    Path(__file__).absolute().parent / "fastflow/autoscaler/protobufs"
_framework_path = \
    Path(__file__).absolute().parent / "fastflow/autoscaler/framework/grpc"
protoc.main([protoc.__file__,
             "-I{}".format(_protobufs_path),
             "--python_out={}".format(_framework_path)] +
            list(find_files("*.proto", _protobufs_path)) +
            ["-I{}".format(_proto_include)])

setup(
    name='fastflow',
    version=_VERSION,
    description='fastflow',
    author='Samsung Research',
    url='https://github.com/SamsungLabs/FastFlow',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(include=['fastflow*']),
    keywords=['fastflow', 'tensorflow', 'offloading'],
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
)
