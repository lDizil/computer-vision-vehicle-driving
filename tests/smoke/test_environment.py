import importlib
import json
from pathlib import Path

import fiona
import numpy as np
import pytest
import torch

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import AgentInput, EgoStatus


def test_editable_upstream_installations():
    for name, source in (
        ("navsim", Path("/opt/navsim/navsim")),
        ("nuplan", Path("/opt/nuplan-devkit/nuplan")),
    ):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(source)


def test_linux_geospatial_stack(tmp_path):
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

    assert GPKGMapsDB.__module__ == "nuplan.database.maps_db.gpkg_mapsdb"
    path = tmp_path / "smoke.gpkg"
    schema = {"geometry": "Point", "properties": {"name": "str"}}
    with fiona.open(path, "w", driver="GPKG", schema=schema, crs="EPSG:4326") as layer:
        layer.write(
            {
                "geometry": {"type": "Point", "coordinates": (12.0, 55.0)},
                "properties": {"name": "smoke"},
            }
        )

    with fiona.open(path) as layer:
        features = list(layer)
        assert len(features) == 1
        assert features[0]["geometry"]["coordinates"] == (12.0, 55.0)
        assert features[0]["properties"]["name"] == "smoke"


def test_cpu_tensor_operation():
    actual = torch.tensor([[1.0, 2.0]]) @ torch.tensor([[3.0], [4.0]])
    assert actual.device.type == "cpu"
    assert actual.item() == 11.0
    assert not torch.cuda.is_available()


@pytest.mark.parametrize(
    ("velocities", "expected_x"),
    [
        ([(0.0, 0.0)], [0.0] * 8),
        ([(3.0, 4.0)], [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]),
        ([(30.0, 40.0), (0.0, 2.0)], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    ],
)
def test_real_constant_velocity_agent(velocities, expected_x):
    agent_input = AgentInput(
        ego_statuses=[
            EgoStatus(
                ego_pose=np.zeros(3, dtype=np.float64),
                ego_velocity=np.array(velocity, dtype=np.float32),
                ego_acceleration=np.zeros(2, dtype=np.float32),
                driving_command=np.array([1, 0, 0], dtype=np.int64),
            )
            for velocity in velocities
        ],
        cameras=[],
        lidars=[],
    )
    agent = ConstantVelocityAgent()
    agent.initialize()
    trajectory = agent.compute_trajectory(agent_input)

    assert trajectory.poses.shape == (8, 3)
    assert np.isfinite(trajectory.poses).all()
    np.testing.assert_allclose(trajectory.poses[:, 0], expected_x)
    np.testing.assert_array_equal(trajectory.poses[:, 1:], np.zeros((8, 2)))
    assert trajectory.trajectory_sampling.time_horizon == 4.0
    assert trajectory.trajectory_sampling.interval_length == 0.5

    print(json.dumps({"agent": agent.name(), "poses": trajectory.poses.tolist()}))
