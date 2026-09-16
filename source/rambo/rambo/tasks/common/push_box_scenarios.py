"""Resolved DATA-006 episode geometry; no changes to the dataset core contract."""
import copy
import math
import numpy as np
from .push_box_geometry import source_geometry,world_geometry,rotation_xyzw


def resolve_episode(base,episode,low,high):
    if episode.get('work_id')!='DATA-006' or episode.get('authorization')!='data006-expansion-v1':
        raise ValueError('Explicit DATA-006 episode authorization required')
    if episode['yaw_deg'] not in (0,90,180,270) or episode['motion'] not in ('coordinated','leg_finish','body_finish'):
        raise ValueError('Unsupported approved episode variant')
    a=episode['parameters']
    for key,lo,hi in [('face_x_m',.52,.565),('center_y_m',.132,.152),('goal_margin_x_m',.1,.2),('goal_margin_y_m',.1,.2),('base_speed_m_s',.1,.14),('transition_s',.3,.6)]:
        if not lo<=a[key]<=hi:raise ValueError('Out-of-range '+key)
    lo,hi=(.36,.38) if episode['motion']=='coordinated' else (.32,.36)
    if not lo<=a['foot_x_m']<=hi:raise ValueError('Out-of-range foot reference')
    angle=math.radians(episode['yaw_deg']);q=[0.,0.,math.sin(angle/2),math.cos(angle/2)]
    R=rotation_xyzw(q);choices=[('local_x_min',[-1.,0.,0.]),('local_x_max',[1.,0.,0.]),('local_y_min',[0.,-1.,0.]),('local_y_max',[0.,1.,0.])]
    face=min(choices,key=lambda item:float((R@item[1])[0]))[0]
    geom=source_geometry(low,high,face=face);zero=world_geometry(geom,[0.,0.,0.,*q]);pts=np.asarray(geom['corners'])@R.T
    position=[a['face_x_m']-min(pts[:,0]),a['center_y_m']-zero['center'][1],base['initial_floor_clearance']-min(pts[:,2])]
    wx=float(np.ptp(pts[:,0]));wy=float(np.ptp(pts[:,1]));gx=max(.4,wx+a['goal_margin_x_m']);gy=max(.5,wy+a['goal_margin_y_m'])
    p=copy.deepcopy(base);p.update(work_id='DATA-006',asset_path=episode['asset_path'],asset_sha256=episode['asset_sha256'],initial_box_yaw_rad=angle,push_face=face,goal_x=[.95,.95+gx],goal_y=[.15-gy/2,.15+gy/2],goal_color=episode['goal_color'],episode_scenario=episode)
    p['episode_scenario']={**episode,'selected_face':face}
    p['expert'].update(push_base_speed=a['base_speed_m_s'],push_foot_x=a['foot_x_m'],motion=episode['motion'],transition_s=a['transition_s'],leg_finish_max_x=.40,lateral_tracking_gain=2.,lateral_tracking_limit_m=.015)
    p['expert']['corner_recovery']=bool(episode.get('corner_recovery',False))
    return p,geom,position,q


def goal_markers(profile):
    x0,x1=profile['goal_x'];y0,y1=profile['goal_y'];cx=(x0+x1)/2;cy=(y0+y1)/2
    return [((x0,cy,.002),(.008,y1-y0,.003)),((x1,cy,.002),(.008,y1-y0,.003)),((cx,y0,.002),(x1-x0,.008,.003)),((cx,y1,.002),(x1-x0,.008,.003))]
