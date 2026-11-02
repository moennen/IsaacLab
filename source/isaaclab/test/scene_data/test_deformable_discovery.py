# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declared deformable prototype classification and clone-plan expansion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_clone_plan
from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.scene_data.deformable_discovery import (
    DeformableStageEntry,
    deformable_entry,
    deformable_geometry_batches,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.scene_data.deformable_gaussian_splats import (
    WarpGaussianSplatsEmbedding,
    _sh_basis,
    read_gaussian_splats_embedding,
)
from isaaclab.sim import SpawnerCfg
from isaaclab.test.utils.devices import test_devices


def _add_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def test_deformable_entry_volume_tet_mesh():
    """Classify tetrahedra, bake sim vertices, and prefer the named visual over unrelated child meshes."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/SoftBody").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])
    tet = UsdGeom.TetMesh.Define(stage, "/World/envs/env_0/SoftBody/simulation")
    _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI"])
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    tet.CreatePointsAttr(points)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])
    # A rotated and translated sim mesh: vertices are baked into the root's parent frame.
    tet.AddTranslateOp().Set(Gf.Vec3d(0.5, -0.1, 0.25))
    tet.AddRotateYOp().Set(30.0)
    for name in ("decoration", "visual", "props/unrelated"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_0/SoftBody/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    entry = deformable_entry(root)
    assert entry.deformable_type == "volume"
    assert entry.vertex_count == 4
    assert entry.vis_vertex_count == 4
    assert entry.root_path.endswith("/SoftBody")
    assert entry.sim_mesh_path.endswith("/simulation")
    assert entry.vis_mesh_path.endswith("/visual")
    # USD's own row-vector transform is the reference for the baked vertices.
    matrix = tet.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    expected = np.array([list(matrix.Transform(Gf.Vec3d(p))) for p in points], dtype=np.float32)
    np.testing.assert_allclose(entry.vertices, expected, atol=1e-6)
    gaussian = stage.DefinePrim("/World/envs/env_0/SoftBody/splats", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.25, 0.25, 0.25)])
    gaussian.CreateAttribute("isaaclab:embeddingOffsets", Sdf.ValueTypeNames.IntArray).Set([0, 4])
    gaussian.CreateAttribute("isaaclab:embeddingIndices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])
    gaussian.CreateAttribute("isaaclab:embeddingWeights", Sdf.ValueTypeNames.FloatArray).Set([0.25] * 4)
    _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI", "PhysicsDeformablePoseAPI:bind"])
    tet.GetPrim().CreateAttribute("physics:deformablePose:bind:purposes", Sdf.ValueTypeNames.TokenArray).Set(
        ["bindPose"]
    )
    tet.GetPrim().CreateAttribute("physics:deformablePose:bind:points", Sdf.ValueTypeNames.Point3fArray).Set(
        [(p[0] * 0.5, p[1], p[2]) for p in points]
    )
    bound = deformable_entry(root)
    assert [embedding.path for embedding in bound.gaussian_splats_embeddings] == [str(gaussian.GetPath())]
    np.testing.assert_allclose(bound.vertices[1], expected[1], atol=1e-6)
    np.testing.assert_allclose(bound.bind_vertices[1], list(matrix.Transform(Gf.Vec3d(0.5, 0, 0))), atol=1e-6)
    tet.GetPrim().GetAttribute("physics:deformablePose:bind:points").Set([(0, 0, 0)])
    with pytest.raises(ValueError, match="Bind pose point count"):
        deformable_entry(root)


@pytest.mark.parametrize("device", test_devices())
def test_gaussian_splats_embedding_reads_authored_weights_and_falls_back_to_bind_pose(device):
    """Asset weights select the intended element; an unbound splat embeds in the bind tet."""
    stage = Usd.Stage.CreateInMemory()
    gaussian = stage.DefinePrim("/Visual", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.25, 0.25, 0.25)])
    points = np.array(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (5, 0, 0), (6, 0, 0), (5, 1, 0), (5, 0, 1)],
        dtype=np.float32,
    )
    elements = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

    derived = read_gaussian_splats_embedding(gaussian, points, elements, element_width=4)
    np.testing.assert_array_equal(derived.indices, elements[:4])
    np.testing.assert_allclose(derived.weights, [0.25] * 4)
    assert derived.channels == ("centers",)

    gaussian.CreateAttribute("isaaclab:embeddingOffsets", Sdf.ValueTypeNames.IntArray).Set([0, 4])
    gaussian.CreateAttribute("isaaclab:embeddingIndices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])
    gaussian.CreateAttribute("isaaclab:embeddingWeights", Sdf.ValueTypeNames.FloatArray).Set([0.7, 0.1, 0.1, 0.1])
    gaussian.CreateAttribute("isaaclab:deformableChannels", Sdf.ValueTypeNames.TokenArray).Set(
        ["centers", "covariance"]
    )
    authored = read_gaussian_splats_embedding(gaussian, points, elements, element_width=4)
    np.testing.assert_allclose(authored.weights, [0.7, 0.1, 0.1, 0.1])
    assert authored.channels == ("centers", "covariance")
    current = points.copy()
    current[:, 0] *= 2.0
    warp_embedding = WarpGaussianSplatsEmbedding.create(authored, points, (0, 0, 0), (0, 0, 0, 1), device)
    state = warp_embedding.evaluate(wp.array(current, dtype=wp.vec3f, device=device))
    np.testing.assert_allclose(state.positions.numpy()[0], [0.2, 0.1, 0.1])
    np.testing.assert_allclose(np.sort(state.scales.numpy()[0]), [1, 1, 2], atol=1e-5)
    device_rotation = Rotation.from_quat(state.orientations.numpy()[0]).as_matrix()
    device_covariance = device_rotation @ np.diag(state.scales.numpy()[0] ** 2) @ device_rotation.T
    np.testing.assert_allclose(device_covariance, np.diag([4, 1, 1]), atol=1e-5)
    gaussian.GetAttribute("isaaclab:embeddingOffsets").Set([0, 3])
    gaussian.GetAttribute("isaaclab:embeddingIndices").Set([0, 1, 2])
    gaussian.GetAttribute("isaaclab:embeddingWeights").Set([1 / 3] * 3)
    with pytest.raises(ValueError, match="degenerate proxy nodes"):
        read_gaussian_splats_embedding(gaussian, points, elements, element_width=4)


def test_particle_covariance_embedding_requires_a_full_rank_neighborhood():
    """A nearest-particle fallback cannot define a Gaussian's local deformation."""
    stage = Usd.Stage.CreateInMemory()
    gaussian = stage.DefinePrim("/Visual", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.1, 0.1, 0.1)])
    gaussian.CreateAttribute("isaaclab:deformableChannels", Sdf.ValueTypeNames.TokenArray).Set(
        ["centers", "covariance"]
    )
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)

    with pytest.raises(ValueError, match="degenerate proxy nodes"):
        read_gaussian_splats_embedding(gaussian, points, np.empty(0, dtype=np.int32), element_width=0)
    gaussian.CreateAttribute("isaaclab:embeddingOffsets", Sdf.ValueTypeNames.IntArray).Set([0, 4])
    gaussian.CreateAttribute("isaaclab:embeddingIndices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])
    gaussian.CreateAttribute("isaaclab:embeddingWeights", Sdf.ValueTypeNames.FloatArray).Set([0.4, 0.2, 0.2, 0.2])
    embedding = read_gaussian_splats_embedding(gaussian, points, np.empty(0, dtype=np.int32), element_width=0)
    np.testing.assert_allclose(embedding.weights, [0.4, 0.2, 0.2, 0.2])


@pytest.mark.parametrize("device", test_devices())
def test_surface_gaussian_splats_embedding_preserves_normal_distance(device):
    """A splat above cloth follows the rotated triangle without collapsing onto its plane."""
    stage = Usd.Stage.CreateInMemory()
    gaussian = stage.DefinePrim("/Visual", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.2, 0.2, 0.3)])
    rest = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
    embedding = read_gaussian_splats_embedding(gaussian, rest, np.array([0, 1, 2]), element_width=3)
    rotated = rest @ Rotation.from_euler("x", 90, degrees=True).as_matrix().T

    warp_embedding = WarpGaussianSplatsEmbedding.create(embedding, rest, (0, 0, 0), (0, 0, 0, 1), device)
    state = warp_embedding.evaluate(wp.array(rotated.astype(np.float32), dtype=wp.vec3f, device=device))
    np.testing.assert_allclose(state.positions.numpy()[0], [0.2, -0.3, 0.2], atol=1e-5)


def test_centers_only_gaussian_splats_keeps_static_covariance_under_a_parent_transform():
    """Static covariance and appearance survive parent and shape transforms."""
    stage = Usd.Stage.CreateInMemory()
    gaussian = stage.DefinePrim("/Visual", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.25, 0.25, 0.25)])
    gaussian.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set([(2, 1, 1)])
    gaussian.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int).Set(1)
    coefficients = np.zeros((4, 3), dtype=np.float32)
    coefficients[3, 0] = 1.0
    gaussian.CreateAttribute("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array).Set(
        [Gf.Vec3f(*map(float, row)) for row in coefficients]
    )
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation.T
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32) @ rotation.T
    embedding = read_gaussian_splats_embedding(
        gaussian, points, np.arange(4), element_width=4, gaussian_splats_to_parent=transform
    )

    warp_embedding = WarpGaussianSplatsEmbedding.create(embedding, points, (0, 0, 0), (0, 0, 0, 1), "cpu")
    state = warp_embedding.evaluate(wp.array(points, dtype=wp.vec3f, device="cpu"))
    frame = Rotation.from_quat(state.orientations.numpy()[0]).as_matrix()
    covariance = frame @ np.diag(state.scales.numpy()[0] ** 2) @ frame.T
    np.testing.assert_allclose(covariance, np.diag([1, 4, 1]), atol=1e-5)
    shape_pose = wp.array([[0, 0, 0, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=wp.transformf, device="cpu")
    local = warp_embedding.to_local(state, shape_pose, wp.array([[1, 1, 1]], dtype=wp.vec3f, device="cpu"), 0)
    world_direction = np.array([[1.0, 0.0, 0.0]])
    local_direction = world_direction @ rotation
    world_color = _sh_basis(world_direction, 1) @ state.sh_coefficients.numpy()
    local_color = _sh_basis(local_direction, 1) @ local.sh_coefficients.numpy()
    np.testing.assert_allclose(local_color, world_color, atol=2e-4)


@pytest.mark.parametrize("device", test_devices())
def test_deformable_gaussian_splats_publication_updates_after_native_nodes_change(device, monkeypatch):
    """The provider evaluates native nodes on their device when physics advances."""
    stage = Usd.Stage.CreateInMemory()
    gaussian = stage.DefinePrim("/Visual", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.25, 0.25, 0.25)])
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
    embedding = read_gaussian_splats_embedding(gaussian, points, np.array([0, 1, 2, 3]), element_width=4)
    entry = DeformableStageEntry(
        "/Body",
        "/Sim",
        "/VisualMesh",
        "volume",
        4,
        4,
        vertices=points,
        vis_vertices=points,
        gaussian_splats_embeddings=[embedding],
    )
    batches = deformable_geometry_batches([entry], [0], device=device)
    assert "/Sim" not in {path for _, paths in batches for path in paths}
    gaussian_splats_only = DeformableStageEntry(
        "/Body",
        "/Sim",
        "/Sim",
        "volume",
        4,
        4,
        vertices=points,
        vis_vertices=points,
        gaussian_splats_embeddings=[embedding],
    )
    assert deformable_geometry_batches([gaussian_splats_only], [0], device=device) == []
    publication = SceneDataFormat.Points()
    publication.points = wp.array(points, dtype=wp.vec3f, device=device)
    reads = []

    def get_gaussian_splats_points():
        reads.append(1)
        return {"/Sim": publication.points}

    backend = SimpleNamespace(
        geometry_timestamp=0,
        get_gaussian_splats_entries=lambda: [entry],
        get_gaussian_splats_points=get_gaussian_splats_points,
        refresh_gaussian_splats_timestamp=lambda: None,
    )
    provider = SceneDataProvider(backend)
    provider.prepare_deformable_gaussian_splats(device)
    reads.clear()
    with monkeypatch.context() as patch:
        patch.setattr(wp.array, "numpy", lambda self: pytest.fail("Gaussian evaluation read back a native array"))
        positions = provider.get_deformable_gaussian_splats()["/Visual"].positions
    assert str(positions.device) == device
    np.testing.assert_allclose(positions.numpy()[0], [0.25] * 3)
    provider.get_deformable_gaussian_splats()
    assert len(reads) == 1
    moved = points + [1, 0, 0]
    publication.points = wp.array(moved, dtype=wp.vec3f, device=device)
    backend.geometry_timestamp += 1
    with monkeypatch.context() as patch:
        patch.setattr(wp.array, "numpy", lambda self: pytest.fail("Gaussian evaluation read back a native array"))
        positions = provider.get_deformable_gaussian_splats()["/Visual"].positions
    np.testing.assert_allclose(positions.numpy()[0], [1.25, 0.25, 0.25])
    assert len(reads) == 2


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("device", test_devices())
def test_directional_gaussian_splats_embedding_rotates_view_dependence(degree, device):
    """A rotated tet turns a directional lobe without changing its constant coefficient."""
    stage = Usd.Stage.CreateInMemory()
    gaussian = stage.DefinePrim("/Visual", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.25, 0.25, 0.25), (0.1, 0.2, 0.3)])
    gaussian.CreateAttribute("isaaclab:deformableChannels", Sdf.ValueTypeNames.TokenArray).Set(
        ["centers", "covariance", "sphericalHarmonics"]
    )
    coefficients = np.zeros((2, (degree + 1) ** 2, 3), dtype=np.float32)
    coefficients[:, 0, 0] = 0.25
    coefficients[0, 3, 0] = 1.0
    coefficients[1, 1, 1] = 0.8
    gaussian.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int).Set(degree)
    gaussian.CreateAttribute("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array).Set(
        [Gf.Vec3f(*map(float, row)) for row in coefficients.reshape(-1, 3)]
    )
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
    embedding = read_gaussian_splats_embedding(gaussian, points, np.array([0, 1, 2, 3]), element_width=4)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    stretch = np.diag([2.0, 1.0, 0.5]).astype(np.float32)
    warp_embedding = WarpGaussianSplatsEmbedding.create(embedding, points, (0, 0, 0), (0, 0, 0, 1), device)
    state = warp_embedding.evaluate(wp.array(points @ stretch @ rotation.T, dtype=wp.vec3f, device=device))
    np.testing.assert_allclose(
        state.positions.numpy(), np.array([[0.25, 0.25, 0.25], [0.1, 0.2, 0.3]]) @ stretch @ rotation.T, atol=1e-5
    )
    updated = state.sh_coefficients.numpy().reshape(coefficients.shape)
    np.testing.assert_allclose(updated[:, 0], coefficients[:, 0], atol=1e-5)
    for index, direction in enumerate(([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])):
        direction = np.asarray(direction)
        original = _sh_basis(direction[None], degree) @ coefficients[index]
        rotated = _sh_basis((rotation @ direction)[None], degree) @ updated[index]
        np.testing.assert_allclose(rotated, original, atol=1e-5)
    gaussian.GetAttribute("isaaclab:deformableChannels").Set(["centers"])
    gaussian.GetAttribute("radiance:sphericalHarmonicsCoefficients").Set([Gf.Vec3f(0.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="coefficients have the wrong size"):
        read_gaussian_splats_embedding(gaussian, points, np.array([0, 1, 2, 3]), element_width=4)


def test_deformable_entry_surface_mesh():
    stage = Usd.Stage.CreateInMemory()
    sim_mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth")
    _add_api_schemas(
        sim_mesh.GetPrim(),
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsSurfaceDeformableSimAPI",
        ],
    )
    sim_points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
    ]
    sim_mesh.CreatePointsAttr(sim_points)
    sim_mesh.CreateFaceVertexCountsAttr([3, 3])
    sim_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    vis_mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth/visual")
    vis_points = sim_points + [Gf.Vec3f(0.5, 0.5, 0.1)]
    vis_mesh.CreatePointsAttr(vis_points)
    vis_mesh.CreateFaceVertexCountsAttr([3, 3])
    vis_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])

    entry = deformable_entry(sim_mesh.GetPrim())
    assert entry.deformable_type == "surface"
    assert entry.vertex_count == 4
    assert entry.vis_vertex_count == 5


def test_deformable_entry_particle_proxy_reads_gaussian_splats_neighborhood():
    """MPM simulation points can drive a Gaussian visual beneath their asset root."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/SoftParticles").GetPrim()
    points = UsdGeom.Points.Define(stage, "/World/SoftParticles/SimulationPoints")
    _add_api_schemas(points.GetPrim(), ["NewtonPointsDeformableSimAPI", "PhysicsDeformableBodyAPI"])
    points.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    UsdGeom.Mesh.Define(stage, "/World/SoftParticles/VisualMesh").CreatePointsAttr([(0, 0, 0)])
    gaussian = stage.DefinePrim("/World/SoftParticles/Splats", "ParticleField3DGaussianSplat")
    gaussian.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.2, 0.2, 0.2)])
    gaussian.CreateAttribute("isaaclab:embeddingOffsets", Sdf.ValueTypeNames.IntArray).Set([0, 4])
    gaussian.CreateAttribute("isaaclab:embeddingIndices", Sdf.ValueTypeNames.IntArray).Set([0, 1, 2, 3])
    gaussian.CreateAttribute("isaaclab:embeddingWeights", Sdf.ValueTypeNames.FloatArray).Set([0.4, 0.2, 0.2, 0.2])

    entry = deformable_entry(root)
    assert entry.deformable_type == "particle"
    assert entry.sim_mesh_path == str(points.GetPath())
    assert entry.vis_mesh_path == entry.sim_mesh_path
    np.testing.assert_allclose(entry.gaussian_splats_embeddings[0].weights, [0.4, 0.2, 0.2, 0.2])


def test_deformable_entry_skips_missing_mesh(caplog):
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/EmptyDeformable").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])

    with caplog.at_level("WARNING", logger="isaaclab.scene_data.deformable_discovery"):
        entry = deformable_entry(root)

    assert entry is None
    assert any("Skipping deformable prim" in record.message for record in caplog.records)


def test_deformable_entry_surface_without_sim_api_uses_first_mesh():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth")
    _add_api_schemas(mesh.GetPrim(), ["OmniPhysicsDeformableBodyAPI"])
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    entry = deformable_entry(mesh.GetPrim())
    assert entry.deformable_type == "surface"
    assert entry.vertex_count == 3
    assert entry.vis_vertex_count == 3


@pytest.mark.parametrize("source_x", [0.0, 10.0])
def test_backend_geometry_nested_owners_partial_rows_and_shared_roots(source_x):
    """Import declared prototypes once; bind exact paths without retaining geometry on the plan."""
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Lab/Cell3")
    parent.AddTranslateOp().Set((source_x, 0.0, 0.0))
    parent.AddRotateZOp().Set(30.0)
    for path, schema in (
        ("/Lab/Cell3/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Lab/Cell3/Nested/Cloth", "PhysicsDeformableBodyAPI"),
        ("/Lab/Cell3/Dormant/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Lab/Cell7/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Shared/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Undeclared/Cloth", "OmniPhysicsDeformableBodyAPI"),
    ):
        mesh = UsdGeom.Mesh.Define(stage, path)
        _add_api_schemas(mesh.GetPrim(), [schema])
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Points.Define(stage, "/Shared/Particles").CreatePointsAttr([(1.0, 2.0, 3.0)])
    mpm_points = UsdGeom.Points.Define(stage, "/Lab/Cell3/SimulationPoints")
    mpm_points.CreatePointsAttr([(1.0, 2.0, 3.0), (0.0, 0.0, 0.0)])
    _add_api_schemas(mpm_points.GetPrim(), ["PhysicsDeformableBodyAPI"])
    sources = ("/Lab/Cell3", "/Lab/Cell3/Nested")
    destinations = ("/Lab/Cell{}", "/Lab/Cell{}/Nested")
    env_ids = np.asarray([3, 7, 11])
    positions = np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [35.0, 0.0, 0.0]])
    shared = ("/Shared", "/Shared/Cloth", "/Lab")
    assets = tuple(
        AssetBaseCfg(prim_path=template.format("[^/]+"), spawn=SpawnerCfg(spawn_path=source))
        for source, template in zip((*sources, *shared), (*destinations, *shared), strict=True)
    )
    plan = make_clone_plan(assets, ((0, 1), (0,), (1,)), 3, shared_assets=(2, 3, 4), env_template="/Lab/Cell{}")
    excluded = ("/Missing", "/Lab/Cell3/Dormant")
    prototypes = deformable_prototypes(stage, plan, exclude_paths=excluded)
    nested = deformable_prototypes(stage, plan, exclude_paths=(*sources[:1], *excluded))
    assert len(prototypes) == 3
    assert {entry.root_path for entry in prototypes} == {"/Lab/Cell3/Cloth", "/Lab/Cell3/Nested/Cloth", "/Shared/Cloth"}
    assert {entry.root_path for entry in nested} == {"/Lab/Cell3/Nested/Cloth", "/Shared/Cloth"}
    prototype = next(entry for entry in prototypes if entry.root_path == "/Lab/Cell3/Cloth")
    stage.RemovePrim("/Lab")

    expanded = {entry.root_path: entry for entry in expand_deformable_entries(prototypes, plan, env_ids, positions)}
    assert set(expanded) == {
        "/Lab/Cell3/Cloth",
        "/Lab/Cell7/Cloth",
        "/Lab/Cell3/Nested/Cloth",
        "/Lab/Cell7/Nested/Cloth",
        "/Lab/Cell11/Nested/Cloth",
        "/Shared/Cloth",
    }
    assert {
        entry.root_path
        for entry in expand_deformable_entries(nested, plan, env_ids, positions, imported_sources={sources[1]})
    } == {
        "/Lab/Cell3/Nested/Cloth",
        "/Lab/Cell11/Nested/Cloth",
        "/Shared/Cloth",
    }
    clone = expanded["/Lab/Cell7/Cloth"]
    assert clone.sim_mesh_path == clone.vis_mesh_path == clone.root_path
    assert clone.vertices is prototype.vertices and clone.indices is prototype.indices
    np.testing.assert_allclose(clone.init_pos, (20.0, 0.0, 0.0))
    np.testing.assert_allclose(clone.init_rot, (0.0, 0.0, np.sin(np.pi / 12), np.cos(np.pi / 12)))

    # Distinct source roots can target the same subtree; its nearest destination owner wins.
    shared = next(entry for entry in prototypes if entry.root_path == "/Shared/Cloth")
    assets = tuple(
        AssetBaseCfg(prim_path=template.format("[^/]+"), spawn=SpawnerCfg(spawn_path=source))
        for source, template in zip(("/Lab/Cell3", "/Shared"), destinations, strict=True)
    )
    plan = make_clone_plan(assets, ((0, 1),), 1, env_template="/Lab/Cell{}")
    for ordered in (prototypes, prototypes[::-1]):
        expanded = {entry.root_path: entry for entry in expand_deformable_entries(ordered, plan, np.asarray([3]))}
        assert expanded["/Lab/Cell3/Nested/Cloth"].vertices is shared.vertices


def test_half_precision_gaussian_splats_preserves_authored_channels():
    """Half-only splat assets retain centers, covariance, and directional appearance."""
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/Splats", "ParticleField3DGaussianSplat")
    prim.CreateAttribute("positionsh", Sdf.ValueTypeNames.Half3Array).Set([(0.25, 0.25, 0.25)])
    prim.CreateAttribute("orientationsh", Sdf.ValueTypeNames.QuathArray).Set(
        [Gf.Quath(np.sqrt(0.5), np.sqrt(0.5), 0, 0)]
    )
    prim.CreateAttribute("scalesh", Sdf.ValueTypeNames.Half3Array).Set([(1, 2, 3)])
    prim.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int).Set(0)
    prim.CreateAttribute("radiance:sphericalHarmonicsCoefficientsh", Sdf.ValueTypeNames.Half3Array).Set(
        [(0.5, 0.25, 0.125)]
    )
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
    embedding = read_gaussian_splats_embedding(prim, points, np.arange(4), element_width=4)
    np.testing.assert_allclose(embedding.bind_centers, [[0.25] * 3])
    np.testing.assert_allclose(embedding.bind_orientations, [[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], atol=5e-4)
    np.testing.assert_allclose(embedding.bind_scales, [[1, 2, 3]])
    np.testing.assert_allclose(embedding.sh_coefficients, [[0.5, 0.25, 0.125]])


@pytest.mark.parametrize("device", test_devices())
def test_cloned_gaussian_splats_batch_keeps_independent_motion(device):
    """Shared embedding tables retain each clone's deformation and native buffer replacement."""
    from dataclasses import replace

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/Splats", "ParticleField3DGaussianSplat")
    prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set([(0.25, 0.25, 0.25)])
    prim.CreateAttribute("isaaclab:deformableChannels", Sdf.ValueTypeNames.TokenArray).Set(
        ["centers", "covariance", "sphericalHarmonics"]
    )
    prim.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int).Set(1)
    coefficients = np.array([[0.25, 0.5, 0.75], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
    prim.CreateAttribute("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array).Set(
        [Gf.Vec3f(*map(float, value)) for value in coefficients]
    )
    points = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32)
    embedding = read_gaussian_splats_embedding(prim, points, np.arange(4), element_width=4)
    entry = DeformableStageEntry(
        "/Body0", "/Sim0", "/Mesh0", "volume", 4, 4, vertices=points, gaussian_splats_embeddings=[embedding]
    )
    clone = replace(
        entry,
        root_path="/Body1",
        sim_mesh_path="/Sim1",
        init_pos=(10, 0, 0),
        gaussian_splats_embeddings=[replace(embedding, path="/CloneSplats")],
    )
    first = points * [2, 1, 1]
    second = points * [1, 3, 1] + [10, 0, 0]
    native = wp.array(np.concatenate((first, second)), dtype=wp.vec3f, device=device)
    backend = SimpleNamespace(
        geometry_timestamp=0,
        get_gaussian_splats_entries=lambda: [entry, clone],
        get_gaussian_splats_points=lambda: {"/Sim0": native[:4], "/Sim1": native[4:]},
        refresh_gaussian_splats_timestamp=lambda: None,
    )
    provider = SceneDataProvider(backend)
    provider.prepare_deformable_gaussian_splats(device)
    states = provider.get_deformable_gaussian_splats()
    assert len(states.batches) == 1
    np.testing.assert_allclose(states["/Splats"].positions.numpy(), [[0.5, 0.25, 0.25]], atol=1e-5)
    np.testing.assert_allclose(states["/CloneSplats"].positions.numpy(), [[10.25, 0.75, 0.25]], atol=1e-5)
    np.testing.assert_allclose(np.sort(states["/Splats"].scales.numpy()), [[1, 1, 2]], atol=1e-5)
    np.testing.assert_allclose(np.sort(states["/CloneSplats"].scales.numpy()), [[1, 1, 3]], atol=1e-5)
    np.testing.assert_allclose(states["/Splats"].sh_coefficients.numpy(), coefficients, atol=2e-4)
    np.testing.assert_allclose(states["/CloneSplats"].sh_coefficients.numpy(), coefficients, atol=2e-4)
    native = wp.array(np.concatenate((first, second)) + [0, 0, 2], dtype=wp.vec3f, device=device)
    backend.geometry_timestamp += 1
    updated = provider.get_deformable_gaussian_splats()
    np.testing.assert_allclose(updated["/Splats"].positions.numpy(), [[0.5, 0.25, 2.25]], atol=1e-5)
    np.testing.assert_allclose(updated["/CloneSplats"].positions.numpy(), [[10.25, 0.75, 2.25]], atol=1e-5)
