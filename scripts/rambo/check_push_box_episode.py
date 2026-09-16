"""Independent GT/trajectory QA; contact remains diagnostic, never pair ground truth."""
import argparse,json,math
from pathlib import Path
import numpy as np
from rambo.tasks.common.push_box_geometry import world_geometry,fully_inside


def check(raw):
    c=json.loads((Path(raw)/'capture.json').read_text());s=json.loads((Path(raw)/'summary.json').read_text())
    if not s.get('passed') or c.get('status')!='success':raise ValueError('Episode did not complete successful runtime/terminal gate')
    g=c['initial_geometry']['source'];profile=c['profile'];rows=[]
    for b in c['boundaries']:
        state=b['state'];w=world_geometry(g,state['task.object.position']+state['task.object.orientation'])
        inside=fully_inside(w['footprint_xy'],profile['goal_x'],profile['goal_y'])
        assert inside==b['task_review']['box_fully_inside']
        fallen=state['observation.state.base.position'][2]<.1 or np.linalg.norm(np.array(state['observation.state.projected_gravity'])-[0,0,-1])>.75
        normal=np.asarray(w['face_normal']);tangent=np.cross([0,0,1],normal);tangent/=np.linalg.norm(tangent)
        foot=np.asarray(b['task_review']['actual_fl_eef_world']);delta=foot-np.asarray(w['face_center'])
        rows.append(dict(simulation_time_ns=b['simulation_time_ns'],**w,actual_fl_eef_world=foot.tolist(),inside=inside,fallen=bool(fallen),success_count=b['task_review']['success_count'],normal_gap=float(delta@normal),tangent_offset=float(delta@tangent)))
    assert len(rows)==len(c['commands'])+1 and all(r['inside'] and not r['fallen'] for r in rows[-3:]) and rows[-1]['success_count']==3
    assert not any(r['fallen'] for r in rows)
    for role in ['ego','task_centric']:
        ids=[b['sensor_frame_ids'][role] for b in c['boundaries']];assert all(y-x==1 for x,y in zip(ids,ids[1:]))
    center=np.array([r['center'] for r in rows]);yaw=np.rad2deg(np.unwrap([r['yaw_rad'] for r in rows]));delta=center[-1]-center[0]
    near=[r for r in rows if abs(r['normal_gap'])<.045]
    flags=[]
    if max(abs(yaw.min()-yaw[0]),abs(yaw.max()-yaw[0]))>25:flags.append('large_yaw_requires_review')
    if abs(delta[1])>.06:flags.append('large_lateral_displacement_requires_review')
    report=dict(passed=True,task_success=True,scenario=c.get('scenario'),work_id=c.get('work_id'),actions=len(c['commands']),boundaries=len(rows),duration_s=s['terminal_simulation_time_ns']/1e9,terminal_before_reset=s['terminal_before_reset'],terminal_footprint=rows[-1]['footprint_xy'],terminal_center=rows[-1]['center'],first_full_policy_ns=next(r['simulation_time_ns'] for r in rows if r['inside']),success_ns=s['terminal_simulation_time_ns'],delta_x_m=float(delta[0]),delta_y_m=float(delta[1]),yaw_min_deg=float(yaw.min()),yaw_max_deg=float(yaw.max()),yaw_final_deg=float(yaw[-1]),lateral_range_m=float(np.ptp(center[:,1])),near_face_tangent_median_m=float(np.median([abs(r['tangent_offset']) for r in near])) if near else None,contact='unknown; geometry metrics are diagnostic only',review_flags=flags,review_flags_are_not_task_success_gates=True)
    expert=[x.get('extensions',{}).get('expert',{}) for x in c['commands']]
    report.update(yaw_initial_deg=float(yaw[0]),yaw_change_deg=float(yaw[-1]-yaw[0]),command_phases=sorted({x.get('phase','unknown') for x in expert}),leg_finish_executed=any(x.get('phase')=='leg_finish' for x in expert))
    return report,rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();report,rows=check(args.raw);args.output.mkdir(parents=True,exist_ok=True);(args.output/'behavior.json').write_text(json.dumps(report,indent=2)+'\n');(args.output/'timeline.json').write_text(json.dumps(rows,separators=(',',':'))+'\n');print(json.dumps(report),flush=True)
