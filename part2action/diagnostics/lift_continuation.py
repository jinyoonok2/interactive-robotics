"""Predeclared controller-handoff probes; physical success limits stay fixed."""
PROFILES = {
    'legacy': {'preserve_closure_gain': False, 'settle_steps': 0, 'lift_steps': 20},
    'preserve_closure_gain': {'preserve_closure_gain': True, 'settle_steps': 0, 'lift_steps': 20},
    'settle_only': {'preserve_closure_gain': False, 'settle_steps': 10, 'lift_steps': 20},
    'slow_lift_only': {'preserve_closure_gain': False, 'settle_steps': 0, 'lift_steps': 40},
    'preserve_and_settle': {'preserve_closure_gain': True, 'settle_steps': 10, 'lift_steps': 20},
}


def last_finger_gain(event):
    values = [float(m['kwargs']['positionGain']) for m in event['motors']
              if len(m['args']) >= 2 and m['args'][1] in [9, 10]]
    if len(values) != 2 or values[0] != values[1] or values[0] <= 0:
        raise ValueError('Need matching positive actual closing gains for both fingers')
    return values[0]


def lift_settings(profile, event):
    settings = dict(PROFILES[profile])
    settings.update(profile=profile, lift_m=.05, hold_steps=10,
                    source_finger_gain=last_finger_gain(event))
    settings['finger_gain'] = settings['source_finger_gain'] if settings['preserve_closure_gain'] else .01
    return settings
