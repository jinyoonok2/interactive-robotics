from diagnostics.lift_continuation import lift_settings


def test_profiles_change_only_predeclared_continuation_factors():
    event={'motors':[{'args':[2,9,2,0.],'kwargs':{'positionGain':.1}},
                     {'args':[2,10,2,0.],'kwargs':{'positionGain':.1}}]}
    baseline=lift_settings('legacy',event)
    preserve=lift_settings('preserve_closure_gain',event)
    assert baseline['finger_gain']==.01 and preserve['finger_gain']==.1
    assert preserve['lift_steps']==baseline['lift_steps']==20
    assert preserve['settle_steps']==baseline['settle_steps']==0
    settle=lift_settings('settle_only',event)
    assert settle['settle_steps']==10 and settle['finger_gain']==.01
    slow=lift_settings('slow_lift_only',event)
    assert slow['lift_steps']==40 and slow['hold_steps']==10 and slow['finger_gain']==.01


def test_preserving_existing_low_gain_is_an_identity_control():
    event={'motors':[{'args':[2,9,2,0.],'kwargs':{'positionGain':.01}},
                     {'args':[2,10,2,0.],'kwargs':{'positionGain':.01}}]}
    baseline=lift_settings('legacy',event);preserve=lift_settings('preserve_closure_gain',event)
    assert baseline['finger_gain']==preserve['finger_gain']
    assert baseline['settle_steps']==preserve['settle_steps']
    assert baseline['lift_steps']==preserve['lift_steps']
