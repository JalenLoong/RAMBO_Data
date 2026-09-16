from types import SimpleNamespace
import numpy as np
from rambo.tasks.common.push_box_corner_recovery import corner_trigger,corner_reference

def fixture():
    profile={'goal_x':[.95,1.35],'goal_y':[-.1,.4],'expert':{'corner_recovery':True}}
    geom={'footprint_xy':[[.96,.1],[1.18,.1],[1.18,.43],[.96,.43]],'center':[1.07,.265,.11]}
    env=SimpleNamespace(cfg=SimpleNamespace(approved_profile=profile),actual_fl_world=lambda:np.array([.97,.26,.11]))
    return env,geom,np.array([.63,.06,.3])

def test_trigger_only_near_full_goal_small_side_overflow():
    e,g,b=fixture();assert corner_trigger(g,e.cfg.approved_profile)
    bad={**g,'footprint_xy':[[.70,.1],[.92,.1],[.92,.43],[.70,.43]]};assert not corner_trigger(bad,e.cfg.approved_profile)
    bad={**g,'footprint_xy':[[.96,.3],[1.18,.3],[1.18,.6],[.96,.6]]};assert not corner_trigger(bad,e.cfg.approved_profile)

def test_brake_then_left_shift_then_inward_stroke():
    e,g,b=fixture();target=np.array([.98,.265,.11])
    _,vx,vy,phase=corner_reference(e,10.,g,b,target,.12);assert phase=='corner_brake' and vx==.12 and vy==0
    _,vx,vy,phase=corner_reference(e,11.,g,b,target,.12);assert phase=='corner_sidestep' and vx==0 and 0<vy<=.05
    b=b+np.array([0,.13,0]);p0,vx,vy,phase=corner_reference(e,14.5,g,b,target,.12)
    p1,vx,vy,phase=corner_reference(e,15.7,g,b,target,.12)
    assert phase=='corner_nudge' and vx==0 and vy==0 and p1[1]<p0[1]
    assert np.isfinite(p1).all() and p1[0]<=b[0]+.42

def test_disabled_path_preserves_command():
    e,g,b=fixture();e.cfg.approved_profile['expert']['corner_recovery']=False;t=np.array([1.,.2,.1])
    out,vx,vy,phase=corner_reference(e,10.,g,b,t,.12);assert np.array_equal(out,t) and vx==.12 and vy==0 and phase is None
