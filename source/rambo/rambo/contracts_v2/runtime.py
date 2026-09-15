"""Pure synchronous protocol and native command ledger; never steps Isaac itself."""
from __future__ import annotations
import copy
import math
import numpy as np
from .validation import (VERSION, CAMERAS, PROFILE_HASHES, BUNDLE, ContractError,
                         require, digest, validate_record, validate_profiles,
                         validate_command, validate_frame)


def prepare_command(requested, command_id, start_tick):
    """Validate all nine values before producing a non-mutating force-zero copy."""
    require(type(command_id) is str and bool(command_id), 'action', 'command_id')
    require(type(start_tick) is int and 0<=start_tick<=2**63-11 and start_tick%10==0, 'timing','command start')
    try:
        if isinstance(requested,(list,tuple)):
            require(not any(isinstance(x,(bool,np.bool_)) for x in requested),'action','boolean is not a physical action')
        raw=np.asarray(requested)
        require(raw.shape==(9,) and raw.dtype.kind in 'fiu', 'action','nine numeric values required')
        raw=raw.astype(np.float64)
        require(np.isfinite(raw).all() and np.max(np.abs(raw))<=np.finfo(np.float32).max,'action','finite float32-representable values required')
    except (ValueError,TypeError,OverflowError) as error:
        if isinstance(error,ContractError):raise
        raise ContractError('action',error) from error
    # Match the existing setter's dtype; preserve the unquantized request above.
    transformed=raw.astype(np.float32).astype(np.float64);transformed[6:]=0
    result=dict(command_id=command_id,start_tick=start_tick,end_tick=start_tick+10,
                executed_until_tick=start_tick,requested=raw.tolist(),transformed=transformed.tolist(),
                filtered=transformed.tolist(),executed=None,status='not_executed',
                force_overridden=bool(np.any(raw[6:]!=0)),transform_id='native9_force_zero_v2',filter_id='identity',extensions={})
    validate_command(result)
    return result


class CommandLedger:
    """Acknowledges backend-confirmed 10ms steps; a full command needs two steps."""
    def __init__(self, actions, chunk_id, start_tick):
        require(type(chunk_id) is str and bool(chunk_id),'action','chunk ID required')
        require(type(actions) is list and len(actions)>0 and len(actions)%16==0,'action','K must be a positive multiple of 16')
        prepared=[prepare_command(a,f'{chunk_id}:{i}',start_tick+10*i) for i,a in enumerate(actions)]
        self.commands=prepared
        self.chunk_id=chunk_id;self.start_tick=start_tick;self.tick=start_tick;self.trace=[]

    @property
    def complete(self):
        return self.tick==self.commands[-1]['end_tick']

    def next_step(self):
        require(not self.complete,'state','chunk completed')
        row=self.commands[(self.tick-self.start_tick)//10]
        return dict(start_tick=self.tick,end_tick=self.tick+5,command_id=row['command_id'],held=copy.deepcopy(row['filtered']))

    def confirm(self, record):
        expected=self.next_step();validate_record('control_step',record)
        require(all(record[k]==v for k,v in expected.items()),'execution','backend confirmation differs from scheduled step')
        require(record['fl_ik_computed_tick']<=self.tick,'timing','future IK computation')
        row=self.commands[(self.tick-self.start_tick)//10]
        self.trace.append(copy.deepcopy(record));self.tick=record['end_tick']
        row['executed_until_tick']=self.tick
        row['status']='completed' if self.tick==row['end_tick'] else 'partial'
        row['executed']=copy.deepcopy(row['filtered']) if row['status']=='completed' else None
        validate_command(row)

    def acknowledgement(self, reason):
        return dict(chunk_id=self.chunk_id,start_tick=self.start_tick,end_tick=self.tick,
                    completed_count=sum(x['status']=='completed' for x in self.commands),
                    commands=copy.deepcopy(self.commands),reason=reason)


class SynchronousSession:
    """Single-environment state machine with externally supplied monotonic wall time.

    cache_resetters are zero-argument adapters for both VAE streams, pending RGB,
    model KV and history. The real adapters/transport are later integration work.
    backend.step(schedule) must return the actual completed control-step record.
    """
    def __init__(self, *, cache_resetters=()):
        self.resetters=tuple(cache_resetters);self.generation=0;self.identity=None
        self.state='NEW';self.tick=0;self.history=[];self.pending=None
        self.responses={};self.chunks=set();self.last_now=None;self.deadline=None
        self.last_observation_tick=None;self.observation_id=None;self.last_error=None
        self.sensor_ids={k:None for k in CAMERAS}

    def _reply(self, kind, body, message_id):
        value=dict(protocol_version=VERSION,session_id=self.identity[0],episode_id=self.identity[1],
                   reset_epoch=self.identity[2],message_id=message_id,profiles=copy.deepcopy(PROFILE_HASHES),
                   extensions={},type=kind,body=body)
        validate_record(kind,value);return value

    def _clock(self, now):
        require(type(now) in (int,float) and math.isfinite(now) and now>=0,'clock','monotonic wall time required')
        require(self.last_now is None or now>=self.last_now,'clock','wall clock reversed')
        self.last_now=now

    def poll(self, *, now):
        self._clock(now)
        if self.state=='WAITING' and now>=self.deadline:
            self.state='ERROR';self.pending=None
            self.last_error=self._reply('Error',dict(tick=self.tick,code='timeout',reason='policy deadline exceeded; simulation remains frozen'),f'timeout:{self.generation}:{self.observation_id}')
        return copy.deepcopy(self.last_error)

    def handle(self, message, *, now):
        require(type(message) is dict and type(message.get('type')) is str,'schema','message type')
        kind=message['type'];validate_record(kind,message);validate_profiles(message['profiles'])
        self._clock(now)
        identity=(message['session_id'],message['episode_id'],message['reset_epoch'])
        if kind!='ResetRequest':
            require(identity==self.identity,'epoch','session/episode/epoch mismatch')
        elif self.identity is not None:
            require(identity==self.identity or identity[0]!=self.identity[0] or identity[2]>self.identity[2], 'epoch','reset epoch must advance within a session')
        key=(*identity,message['message_id']);fingerprint=digest(message)
        if key in self.responses:
            old,result=self.responses[key]
            require(old==fingerprint,'duplicate','same message ID with different contents')
            return copy.deepcopy(result)
        body=message['body']
        if kind=='ResetRequest':
            require(identity!=self.identity,'epoch','new reset requires a new epoch')
            self.state='ERROR'
            try:
                for reset in self.resetters:reset()
            except Exception as error:
                raise ContractError('cache_reset',error) from error
            self.identity=identity;self.generation+=1;self.tick=0;self.history=[];self.pending=None
            self.responses={};self.chunks=set();self.deadline=None;self.last_error=None
            self.last_observation_tick=None;self.observation_id=None;self.sensor_ids={k:None for k in CAMERAS}
            self.timeout=body['timeout_s'];self.task=copy.deepcopy(body['task']);self.origin_ns=body['origin_sim_ns']
            self.state='READY';result=self._reply('Ready',dict(tick=0,cache_generation=self.generation),'reply:'+message['message_id'])
        elif kind=='Observation':
            require(self.state in ('READY','AWAIT_OBSERVATION'),'state','not awaiting observation')
            require(body['tick']==self.tick,'timing','observation cannot advance simulation')
            require(body['instruction']==self.task['instruction'],'task','instruction changed without reset')
            require(body['executed_history']==self.history,'history','history must exactly match confirmed commands')
            previous=-40 if self.last_observation_tick is None else self.last_observation_tick
            expected=list(range(previous+40,self.tick+1,40));seen={k:[] for k in CAMERAS};last_ids=dict(self.sensor_ids)
            require(expected,'camera','no new camera capture')
            for frame in body['rgb']:
                validate_frame(frame,epoch=identity[2],origin_ns=self.origin_ns,last_tick=self.tick)
                k=frame['camera_id'];seen[k].append(frame['capture_tick'])
                require(last_ids[k] is None or frame['sensor_frame_id']>last_ids[k],'camera','stale sensor frame')
                last_ids[k]=frame['sensor_frame_id']
            require(all(seen[k]==expected for k in CAMERAS),'timing','RGB increments missing or out of order')
            deadline=now+self.timeout
            require(math.isfinite(deadline),'clock','deadline overflow')
            self.sensor_ids=last_ids;self.last_observation_tick=self.tick;self.observation_id=message['message_id']
            self.state='WAITING';self.deadline=deadline;result=copy.deepcopy(message)
        elif kind=='ActionChunk':
            self.poll(now=now);require(self.state=='WAITING','state','not awaiting policy action')
            require(body['observation_id']==self.observation_id,'observation','stale observation')
            require(body['start_tick']==self.tick,'timing','overlapping or skipped action interval')
            require(self.origin_ns+(self.tick+10*len(body['actions']))*2_000_000<=2**63-1,'timing','timestamp overflow')
            require(body['chunk_id'] not in self.chunks,'duplicate','chunk ID reused')
            pending=CommandLedger(body['actions'],body['chunk_id'],self.tick)
            self.pending=pending;self.chunks.add(body['chunk_id']);self.state='EXECUTING';self.deadline=None
            result=self._reply('ActionAccepted',dict(chunk_id=body['chunk_id'],start_tick=self.tick,command_count=len(pending.commands)),'reply:'+message['message_id'])
        elif kind=='End':
            require(self.state in ('READY','AWAIT_OBSERVATION','WAITING'),'state','finish pending execution before End')
            require(body['tick']==self.tick and not(body['terminated'] and body['truncated']),'timing','ending')
            self.state='ENDED';self.deadline=None;result=copy.deepcopy(message)
        else:
            raise ContractError('state',f'{kind} is an output-only message')
        self.responses[key]=(fingerprint,copy.deepcopy(result))
        return copy.deepcopy(result)

    def step(self, backend):
        require(self.state=='EXECUTING' and self.pending is not None,'state','no accepted execution')
        scheduled=self.pending.next_step()
        try:
            record=backend.step(copy.deepcopy(scheduled))
            known={r['command_id']:r for r in self.history+self.pending.commands}
            validate_record('control_step',record)
            for k in ('residual_observation_command_id','fl_ik_command_id'):
                ident=record[k]
                require(ident is None or (ident in known and known[ident]['start_tick']<=self.tick),'execution','unknown/future consumer command')
            if record['fl_ik_command_id'] is not None:
                require(known[record['fl_ik_command_id']]['start_tick']<=record['fl_ik_computed_tick'],'timing','IK precedes its source command')
            self.pending.confirm(record)
        except Exception as error:
            self.state='ERROR'
            if isinstance(error,ContractError):raise
            raise ContractError('backend',error) from error
        self.tick=self.pending.tick
        if self.pending.complete:
            return self.finish(reason='completed')
        return copy.deepcopy(record)

    def finish(self, *, reason):
        require(self.pending is not None and self.state in ('EXECUTING','ERROR'),'state','no pending chunk')
        require(type(reason) is str and reason,'execution','reason required')
        ledger=self.pending
        require(reason!='completed' or ledger.complete,'execution','incomplete chunk')
        self.history.extend(copy.deepcopy([r for r in ledger.commands if r['status']=='completed']))
        self.tick=ledger.tick;self.pending=None
        self.state='AWAIT_OBSERVATION' if ledger.complete and reason=='completed' else 'ENDED'
        return self._reply('ExecutionAck',ledger.acknowledgement(reason),f'ack:{self.generation}:{ledger.chunk_id}')
