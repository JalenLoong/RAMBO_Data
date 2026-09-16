"""Dimension-derived push face and full-footprint predicate; CPU-only geometry."""
from itertools import product
import math
import numpy as np


def rotation_xyzw(q):
    x,y,z,w=np.asarray(q,dtype=np.float64)
    if not np.isclose(x*x+y*y+z*z+w*w,1.,atol=1e-4):raise ValueError('Unit XYZW quaternion required')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def source_geometry(low,high):
    low=np.asarray(low,dtype=np.float64);high=np.asarray(high,dtype=np.float64)
    if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or not (high>low).all():raise ValueError('Invalid mesh bounds')
    center=(low+high)/2
    # Preserve the reviewed nominal yaw. The face opposing +world X is x-min.
    face=center.copy();face[0]=low[0]
    return dict(low=low.tolist(),high=high.tolist(),dimensions=(high-low).tolist(),center=center.tolist(),
                face_center=face.tolist(),face_normal=[-1.,0.,0.],face='local_x_min',
                corners=np.array(list(product(*zip(low,high)))).tolist())


def world_geometry(geometry,pose):
    pose=np.asarray(pose,dtype=np.float64);R=rotation_xyzw(pose[3:]);origin=pose[:3]
    vertices=np.asarray(geometry['corners'])@R.T+origin
    low=vertices[:,:2].min(0);high=vertices[:,:2].max(0)
    # Four corners of the projected cuboid's XY envelope. For an axis-aligned
    # rectangular goal this is equivalent to containing all eight 3D corners,
    # and includes roll/pitch projection instead of pretending the box is flat.
    footprint=[[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]]
    return dict(center=(R@geometry['center']+origin).tolist(),face_center=(R@geometry['face_center']+origin).tolist(),
                face_normal=(R@geometry['face_normal']).tolist(),footprint_xy=footprint,
                projected_box_corners=vertices[:,:2].tolist(),yaw_rad=math.atan2(R[1,0],R[0,0]))


def fully_inside(footprint,goal_x,goal_y):
    points=np.asarray(footprint,dtype=np.float64)
    return bool(np.all(points[:,0]>=goal_x[0]) and np.all(points[:,0]<=goal_x[1]) and
                np.all(points[:,1]>=goal_y[0]) and np.all(points[:,1]<=goal_y[1]))


class PolicySuccessHold:
    def __init__(self,required_ticks=3):
        if required_ticks not in (2,3):raise ValueError('Use two or three policy ticks')
        self.required=required_ticks;self.reset()
    def reset(self):
        self.count=0;self.last_tick=None;self.first_inside_ns=None
    def update(self,physics_tick,inside,not_fallen):
        if physics_tick%10 or physics_tick==self.last_tick:return self.count>=self.required and not_fallen
        if self.last_tick is not None and physics_tick!=self.last_tick+10:self.count=0
        self.last_tick=physics_tick
        if inside and self.first_inside_ns is None:self.first_inside_ns=physics_tick*2_000_000
        self.count=self.count+1 if inside and not_fallen else 0
        return self.count>=self.required and not_fallen
