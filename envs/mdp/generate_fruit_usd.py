#!/usr/bin/env python3
"""
One-time, OFFLINE generator: builds the apple/pear lathe mesh and saves it
to a standalone .usd file. Run this ONCE, outside of training/teleop.

Needs only the `pxr` USD python bindings (`pip install usd-core
--break-system-packages`) -- NOT Isaac Sim / a GPU / a running kit process.
That's why this works fine even in a plain python environment: building
and saving a USD mesh file is just authoring data, no physics simulation
involved yet.

After running this, point your RigidObjectCfg.spawn at the resulting file:

    apple: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Apple",
        spawn=sim_utils.UsdFileCfg(
            usd_path="usd/fruit_apple_generated.usd",
            ...
        ),
        ...
    )

exactly the same way G1_DEX3_USD_PATH points at the robot's USD file.
Physics (RigidBodyAPI / CollisionAPI / MassAPI) are already authored into
the file, so Isaac Lab's normal spawn pipeline picks them up like any other
asset -- no per-reset prim surgery needed for the SHAPE. Mass still gets
overwritten every reset by randomize_fruit_scale_and_mass() in fruit_utils.py.
"""

from pxr import Usd, UsdGeom, UsdPhysics, Gf

from fruit_utils import FRUIT_PRESETS, build_lathe_mesh, fruit_volume


def export_fruit_usd(out_path: str, profile_name: str,
                      n_radial: int = 28, n_height: int = 40) -> None:
    profile = FRUIT_PRESETS[profile_name]
    points, face_counts, face_indices = build_lathe_mesh(profile, n_radial, n_height)

    stage = Usd.Stage.CreateNew(out_path)
    mesh = UsdGeom.Mesh.Define(stage, f"/{profile_name}")
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_indices)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.08, 0.08)
                                  if profile_name == "apple" else
                                  Gf.Vec3f(0.72, 0.78, 0.20)])

    prim = mesh.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_col.GetApproximationAttr().Set("convexHull")
    # placeholder mass -- overwritten every reset by randomize_fruit_scale_and_mass
    UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(0.2)

    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().Save()

    v_cm3 = fruit_volume(profile) * 1e6
    print(f"[{profile_name}] saved -> {out_path}  "
          f"(verts={len(points)}, faces={len(face_counts)}, nominal volume={v_cm3:.1f} cm^3)")


if __name__ == "__main__":
    import os
    os.makedirs("generated_usd", exist_ok=True)
    export_fruit_usd("generated_usd/fruit_apple_generated.usd", "apple")
    export_fruit_usd("generated_usd/fruit_pear_generated.usd", "pear")