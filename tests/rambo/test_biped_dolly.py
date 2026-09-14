from pathlib import Path
import numpy as np
import pytest
from rambo.biped_teleop import BipedKeyboardState,DEFAULT_FEET,LOWER,UPPER


def test_independent_front_legs():
    k=BipedKeyboardState();k.event("KEY_1",True);k.event("W",True)
    a=k.advance()
    assert a[3]>DEFAULT_FEET[0,0]
    np.testing.assert_array_equal(a[6:9],DEFAULT_FEET[1])
    assert np.all(a[9:]==0)
    k.event("W",False);k.event("KEY_2",True);k.event("F",True)
    b=k.advance()
    assert b[8]<a[8] and b[5]==a[5]


def test_release_stop_reset_and_bounds():
    k=BipedKeyboardState();k.event("UP",True)
    assert k.advance()[0]>0
    k.event("UP",False);assert k.advance()[0]==0
    k.event("W",True);k.event("F",True)
    for _ in range(1000):k.advance()
    assert np.all(k.feet>=LOWER) and np.all(k.feet<=UPPER)
    k.event("X",True);old=k.feet.copy();k.advance();np.testing.assert_array_equal(k.feet,old)
    k.event("BACKSPACE",True);assert k.reset_requested
    k.reset();np.testing.assert_array_equal(k.feet,DEFAULT_FEET)


def test_passive_dolly_structure():
    usd=pytest.importorskip("pxr.Usd");physics=pytest.importorskip("pxr.UsdPhysics")
    workspace=Path(__file__).resolve().parents[4]
    path=workspace/"assets/dolly/variants/dolly_passive_v1.usda"
    if not path.exists():pytest.skip("Dolly variant not installed")
    stage=usd.Stage.Open(str(path))
    joints=[p for p in stage.Traverse() if p.IsA(physics.RevoluteJoint)]
    bodies=[p for p in stage.Traverse() if p.HasAPI(physics.RigidBodyAPI)]
    assert len(joints)==8 and len(bodies)==9
    assert sum(physics.MassAPI(p).GetMassAttr().Get() for p in bodies)==pytest.approx(8.)
    for joint in joints:
        assert not any("DriveAPI" in s for s in joint.GetAppliedSchemas())
        assert physics.RevoluteJoint(joint).GetBody0Rel().GetTargets()
        assert physics.RevoluteJoint(joint).GetBody1Rel().GetTargets()
