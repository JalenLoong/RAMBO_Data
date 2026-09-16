import numpy as np
import pytest
from rambo.tasks.common.push_box_scenarios import resolve_episode,goal_markers
from rambo.tasks.common.push_box_geometry import world_geometry

@pytest.mark.parametrize('yaw',[0,90,180,270])
def test_face_and_goal_are_resolved_in_world_coordinates(yaw):
    base={'initial_floor_clearance':.02,'expert':{}}
    e=dict(work_id='DATA-006',authorization='data006-expansion-v1',yaw_deg=yaw,motion='coordinated',asset_path='box',asset_sha256='x',goal_color=[0,1,0],parameters=dict(face_x_m=.54,center_y_m=.142,goal_margin_x_m=.15,goal_margin_y_m=.15,base_speed_m_s=.12,foot_x_m=.37,transition_s=.4))
    p,g,pos,q=resolve_episode(base,e,[-.2,-.3,-.1],[.3,.2,.1]);w=world_geometry(g,[*pos,*q])
    assert np.allclose(w['face_normal'],[-1,0,0]);assert abs(w['face_center'][0]-.54)<1e-9
    assert abs(w['center'][1]-.142)<1e-9
    assert abs(min(x[0] for x in w['footprint_xy'])-.54)<1e-9
    m=goal_markers(p);assert m[0][0][0]==p['goal_x'][0] and m[1][0][0]==p['goal_x'][1]
    assert p['goal_x'][1]-p['goal_x'][0]>=.65-1e-9

@pytest.mark.parametrize('q',[[0,0,0,1],[0,0,2**-.5,2**-.5],[0,2**-.5,0,2**-.5]])
def test_center_ray_handles_a_face_becoming_parallel(q):
    from rambo.tasks.common.push_box_geometry import source_geometry,forward_face_ray,rotation_xyzw
    g=source_geometry([-.15,-.09,-.1],[.15,.09,.1]);pose=[1.,.15,.2,*q];point,normal=forward_face_ray(g,pose)
    assert np.isfinite(point).all() and np.isfinite(normal).all();assert np.allclose(point[1:],[.15,.2]);assert point[0]<1.
    local=rotation_xyzw(q).T@(point-pose[:3]);assert np.all(np.abs(local)<=[.1500001,.0900001,.1000001]);assert np.isclose(max(abs(local)/np.array([.15,.09,.1])),1.)
