import numpy as np
from rambo.tasks.common.push_box_geometry import source_geometry,world_geometry,fully_inside,PolicySuccessHold


def test_center_inside_is_not_full_footprint_success():
    g=source_geometry([-.15,-.09,-.1],[.15,.09,.1]);w=world_geometry(g,[.96,.15,.1,0,0,0,1])
    assert .95<w['center'][0]<1.35
    assert not fully_inside(w['footprint_xy'],[.95,1.35],[-.1,.4])
    w=world_geometry(g,[1.15,.15,.1,0,0,0,1])
    assert fully_inside(w['footprint_xy'],[.95,1.35],[-.1,.4])


def test_rotated_and_tilted_corner_envelope_is_conservative():
    g=source_geometry([-.15,-.09,-.11],[.15,.09,.11])
    q=np.array([.12,.2,.35,.9]);q/=np.linalg.norm(q)
    w=world_geometry(g,[1.12,.15,.12,*q]);corners=np.array(w['projected_box_corners'])
    goalx=[.95,1.35];goaly=[-.1,.4]
    allcorners=bool(((corners[:,0]>=goalx[0])&(corners[:,0]<=goalx[1])&(corners[:,1]>=goaly[0])&(corners[:,1]<=goaly[1])).all())
    assert fully_inside(w['footprint_xy'],goalx,goaly)==allcorners


def test_face_reference_comes_from_asset_bounds():
    g=source_geometry([-.142,-.168,-.081],[.159,.007,.146])
    assert np.allclose(g['face_center'],[-.142,(-.168+.007)/2,(-.081+.146)/2])
    assert g['face_normal']==[-1.,0.,0.]
    moved=world_geometry(g,[2.,3.,4.,0,0,0,1])
    assert np.allclose(moved['face_center'],np.array(g['face_center'])+[2,3,4])


def test_hold_counts_unique_consecutive_policy_ticks_only():
    h=PolicySuccessHold(3)
    assert not h.update(10,True,True)
    assert not h.update(10,True,True)
    assert not h.update(15,True,True)
    assert not h.update(20,True,True)
    assert h.update(30,True,True)
    assert not h.update(40,True,False)
    assert h.count==0
    assert not h.update(50,True,True)
    assert not h.update(70,True,True) # missed policy tick resets streak
    assert h.count==1
