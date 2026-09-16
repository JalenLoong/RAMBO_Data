"""Optional near-goal lateral repair through high-level native9 commands only."""
import numpy as np


def smooth(t):
    u=float(np.clip(t,0.,1.));return u*u*(3-2*u)


def corner_trigger(geom,profile):
    pts=np.asarray(geom['footprint_xy']);lo=pts.min(0);hi=pts.max(0)
    gx=profile['goal_x'];gy=profile['goal_y']
    overlap=max(0.,min(hi[1],gy[1])-max(lo[1],gy[0]))/(hi[1]-lo[1])
    excess=hi[1]-gy[1]
    # These reviewed failures overflow on robot-left (+worldY). Only this
    # bounded side is enabled; do not turn a distant/missed goal into repair.
    return bool(lo[0]>=gx[0]+.002 and hi[0]<=gx[1]-.005 and
                .001<excess<=.07 and overlap>=.78)


def corner_reference(env,time,geom,base,target,vx):
    profile=env.cfg.approved_profile
    if not profile['expert'].get('corner_recovery',False):return target,vx,0.,None
    state=getattr(env,'_corner_recovery',None)
    if state is None:
        if not corner_trigger(geom,profile):return target,vx,0.,None
        actual=np.asarray(env.actual_fl_world(),dtype=float)
        state=dict(start=time,base_start=np.asarray(base).copy(),foot_start=actual.copy(),vx_start=float(vx),cycle=-1)
        env._corner_recovery=state
    elapsed=time-state['start'];pts=np.asarray(geom['footprint_xy']);center=np.asarray(geom['center']);upper=float(pts[:,1].max());near=float(pts[:,0].min())
    if elapsed<.4:
        u=smooth(elapsed/.4);end=state['foot_start']+np.array([-.035,0.,.045])
        return state['foot_start']*(1-u)+end*u,state['vx_start']*(1-u),0.,'corner_brake'
    if elapsed<3.6:
        u=smooth((elapsed-.4)/3.2);start=state['foot_start']+np.array([-.035,0.,.045])
        end=np.array([min(near-.025,base[0]+.40),min(upper+.03,base[1]+.32),min(center[2]+.05,.26)])
        remaining=state['base_start'][1]+.16-base[1]
        vy=float(np.clip(1.2*remaining,0.,.05))*smooth((elapsed-.4)/.4)*smooth((3.6-elapsed)/.4)
        return start*(1-u)+end*u,0.,vy,'corner_sidestep'
    # Up to three slow side strokes; all may end early on unchanged success.
    age=elapsed-3.6;cycle=min(int(age/2.2),2);local=age-cycle*2.2
    if state['cycle']!=cycle:
        state.update(cycle=cycle,stroke_start=np.asarray(env.actual_fl_world(),dtype=float),side=np.array([min(center[0],base[0]+.42),min(upper+.025,base[1]+.32),max(.06,center[2])]),inset=float(np.clip(upper-profile['goal_y'][1]+.02,.02,.085)))
    side=state['side'];start=state['stroke_start']
    if local<.5:
        u=smooth(local/.5);raised=side+np.array([0.,0.,.045]);return start*(1-u)+raised*u,0.,0.,'corner_reposition'
    if local<.9:
        u=smooth((local-.5)/.4);return side+np.array([0.,0.,.045*(1-u)]),0.,0.,'corner_lower'
    u=smooth((local-.9)/1.3);goal=side.copy();goal[1]-=state['inset']+.025
    return side*(1-u)+goal*u,0.,0.,'corner_nudge'
