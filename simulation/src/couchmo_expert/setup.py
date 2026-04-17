from pathlib import Path
from glob import glob

from setuptools import setup

package_name = "couchmo_expert"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
            [str(Path("resource") / package_name)]),
        (str(Path("share") / package_name), ["package.xml"]),
        (str(Path("share") / package_name / "launch"), glob("launch/*.py")),
        (str(Path("share") / package_name / "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CouchMo Maintainer",
    maintainer_email="couchmo@example.com",
    description="CouchMo learned-policy support nodes: waypoint expert, action adapter, dataset recorder.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "waypoint_expert_node = couchmo_expert.waypoint_expert_node:main",
            "steer_throttle_to_cmd_vel = couchmo_expert.steer_throttle_to_cmd_vel:main",
        ],
    },
)
