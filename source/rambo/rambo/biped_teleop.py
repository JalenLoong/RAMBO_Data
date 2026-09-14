"""CPU-only keyboard state for native biped base/FL/FR commands."""
import numpy as np

DEFAULT_FEET=np.array([[.2175,.1225,.488],[.2175,-.1225,.488]],np.float32)
LOWER=np.array([[.15,0.,.30],[.15,-.25,.30]],np.float32)
UPPER=np.array([[.35,.25,.65],[.35,0.,.65]],np.float32)
HELP="""Click the viewport to control Go2.
Arrows: move base   Q/E: turn
1: left foot   2: right foot   3: both feet (default)
W/S: foot forward/back   A/D: foot left/right   R/F: foot up/down
X: stop commands   Backspace: reset Go2 and Dolly   Esc: quit
Force commands are zero. Space remains the Kit timeline shortcut."""


class BipedKeyboardState:
    def __init__(self):
        self.quit_requested=False
        self.reset_requested=False
        self.reset()

    def reset(self):
        self.held=set();self.selected=(0,1);self.feet=DEFAULT_FEET.copy()
        self.reset_requested=False

    def event(self,key,pressed):
        if not pressed:
            self.held.discard(key);return
        selection={"1":(0,),"KEY_1":(0,),"2":(1,),"KEY_2":(1,),"3":(0,1),"KEY_3":(0,1)}
        if key in selection:self.selected=selection[key]
        elif key=="X":self.held.clear()
        elif key=="BACKSPACE":self.reset_requested=True;self.held.clear()
        elif key=="ESCAPE":self.quit_requested=True;self.held.clear()
        else:self.held.add(key)

    def advance(self,dt=.01):
        h=self.held
        velocity=np.array([.10*(("UP" in h)-("DOWN" in h)),.08*(("LEFT" in h)-("RIGHT" in h)),
                           .15*(("Q" in h)-("E" in h))],np.float32)
        delta=.08*dt*np.array([("W" in h)-("S" in h),("A" in h)-("D" in h),("R" in h)-("F" in h)])
        for leg in self.selected:self.feet[leg]=np.clip(self.feet[leg]+delta,LOWER[leg],UPPER[leg])
        return np.r_[velocity,self.feet.reshape(-1),np.zeros(6)].astype(np.float32)
