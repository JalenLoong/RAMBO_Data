#!/usr/bin/env python3
"""Author passive physics on a separate NVIDIA Dolly USD variant, offline."""
import argparse
import hashlib
import json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf


def build(source: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    original = Usd.Stage.Open(str(source))
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Dolly").GetPrim()
    stage.SetDefaultPrim(root)
    import os
    relative = os.path.relpath(source, output.parent)
    looks = stage.DefinePrim("/Dolly/Looks", "Scope")
    looks.GetReferences().AddReference(relative, "/Root/Looks")
    visual_material = UsdShade.Material(stage.GetPrimAtPath("/Dolly/Looks/OmniPBR"))
    floor_material = UsdShade.Material.Define(stage, "/Dolly/WheelMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(floor_material.GetPrim())
    material_api.CreateStaticFrictionAttr(.8)
    material_api.CreateDynamicFrictionAttr(.7)
    material_api.CreateRestitutionAttr(0.)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    meshes = {p.GetName(): p for p in original.Traverse() if p.IsA(UsdGeom.Mesh)}
    main = meshes["FOF_Mesh_Shelf_Cart_B_LOD0"]
    bbox = cache.ComputeWorldBound(main).ComputeAlignedRange()
    chassis_center = (bbox.GetMin()+bbox.GetMax())*.5

    def body(name, center, mass):
        prim = UsdGeom.Xform.Define(stage, "/Dolly/"+name).GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(center)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        return prim

    def visual(parent, mesh, center, collision=False):
        p = stage.DefinePrim(str(parent.GetPath())+"/Visual", "Mesh")
        p.GetReferences().AddReference(relative, mesh.GetPath())
        xf = UsdGeom.Xformable(p); xf.ClearXformOpOrder()
        transform = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xf.AddTransformOp(opSuffix="bodyLocal").Set(transform*Gf.Matrix4d().SetTranslate(-center))
        UsdShade.MaterialBindingAPI.Apply(p).Bind(visual_material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(p)
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
        return p

    def joint(name, parent, child, axis, pivot, center0, center1):
        j = UsdPhysics.RevoluteJoint.Define(stage, "/Dolly/Joints/"+name)
        j.CreateBody0Rel().SetTargets([parent.GetPath()])
        j.CreateBody1Rel().SetTargets([child.GetPath()])
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*(pivot-center0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(*(pivot-center1)))
        j.CreateCollisionEnabledAttr(False)

    chassis = body("Chassis", chassis_center, 6.6)
    UsdPhysics.ArticulationRootAPI.Apply(chassis)
    visual(chassis, main, chassis_center, collision=True)
    records=[]
    for n in (1,2,3,4):
        fork_mesh=meshes[f"FOF_Shelf_Cart_B_Swiveling_Wheel{n}_LOD0"]
        wheel_mesh=meshes[f"FOF_Shelf_Cart_B_wheel{n}_LOD0"]
        pivot=UsdGeom.Xformable(fork_mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        bounds=cache.ComputeWorldBound(wheel_mesh).ComputeAlignedRange()
        center=(bounds.GetMin()+bounds.GetMax())*.5
        size=bounds.GetMax()-bounds.GetMin(); radius=float(size[2]/2); width=float(size[1])
        fork=body(f"Caster{n}",pivot,.15);visual(fork,fork_mesh,pivot,collision=True)
        wheel=body(f"Wheel{n}",center,.2);visual(wheel,wheel_mesh,center)
        collider=UsdGeom.Cylinder.Define(stage,str(wheel.GetPath())+"/Collision")
        collider.CreateAxisAttr("Y");collider.CreateRadiusAttr(radius);collider.CreateHeightAttr(width)
        collider.CreatePurposeAttr("guide");collider.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(floor_material,materialPurpose="physics")
        joint(f"CasterYaw{n}",chassis,fork,"Z",pivot,chassis_center,pivot)
        joint(f"WheelRoll{n}",fork,wheel,"Y",center,pivot,center)
        records.append({"number":n,"radius_m":radius,"width_m":width,"wheel_center":list(center),"caster_pivot":list(pivot)})
    stage.GetRootLayer().Save()
    manifest={"asset":"NVIDIA Dolly passive physics v1","source_path":str(source.resolve()),
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"usd_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "total_mass_kg":8.,"chassis_center":list(chassis_center),"rigid_bodies":9,"passive_revolute_joints":8,
        "actuated_joints":0,"wheel_friction":{"static":.8,"dynamic":.7},"wheels":records,
        "collision":"chassis/fork convex hulls; analytic wheel cylinders","source_visuals_preserved":True}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();build(a.source,a.output)
