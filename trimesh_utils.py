from typing import Dict, List

import numpy as np
import sapien.core as sapien
import trimesh


def get_actor_meshes(actor):
    """Get actor (collision) meshes in the actor frame."""
    meshes = []
    for geom in actor.get_collision_shapes():
        if isinstance(geom, sapien.pysapien.physx.PhysxCollisionShapeBox):
            mesh = trimesh.creation.box(extents=2 * geom.half_lengths)
        elif isinstance(geom, sapien.pysapien.physx.PhysxCollisionShapeCapsule):
            mesh = trimesh.creation.capsule(
                height=2 * geom.half_length, radius=geom.radius
            )
        elif isinstance(geom,  sapien.pysapien.physx.PhysxCollisionShapeSphere):
            mesh = trimesh.creation.icosphere(radius=geom.radius)
        elif isinstance(geom, sapien.pysapien.physx.PhysxCollisionShapePlane):
            continue
        elif isinstance(
            geom, (sapien.pysapien.physx.PhysxCollisionShapeConvexMesh, sapien.pysapien.physx.PhysxCollisionShapeTriangleMesh)
        ):
            vertices = geom.get_vertices()  # Retrieves vertex positions as [m, 3] array
            triangles = geom.get_triangles()  # Retrieves indices for triangles as [n, 3] array
            scale = geom.get_scale()  # Retrieves scale as a [3] array
            vertices = vertices * scale
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        else:
            raise TypeError(type(geom))
        mesh.apply_transform(geom.get_local_pose().to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_visual_body_meshes(visual_body):
    meshes = []
    for render_shape in visual_body.get_render_shapes():
        vertices = render_shape.mesh.vertices * visual_body.scale  # [n, 3]
        faces = render_shape.mesh.indices.reshape(-1, 3)  # [m * 3]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.apply_transform(visual_body.local_pose.to_transformation_matrix())
        meshes.append(mesh)
    return meshes


def get_actor_visual_meshes(actor):
    """Get actor (visual) meshes in the actor frame."""
    meshes = []
    for vb in actor.get_visual_bodies():
        meshes.extend(get_visual_body_meshes(vb))
    return meshes


def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None


def get_actor_mesh(actor, to_world_frame=True):
    mesh = merge_meshes(get_actor_meshes(actor))
    if mesh is None:
        return None
    if to_world_frame:
        T = actor.pose.to_transformation_matrix()
        mesh.apply_transform(T)
    return mesh


def get_actor_visual_mesh(actor):
    mesh = merge_meshes(get_actor_visual_meshes(actor))
    if mesh is None:
        return None
    return mesh


def get_articulation_meshes(
    articulation, exclude_link_names=()
):
    """Get link meshes in the world frame."""
    meshes = []
    for link in articulation.get_links():
        if link.name in exclude_link_names:
            continue
        mesh = get_actor_mesh(link, True)
        if mesh is None:
            continue
        meshes.append(mesh)
    return merge_meshes(meshes)
    return meshes