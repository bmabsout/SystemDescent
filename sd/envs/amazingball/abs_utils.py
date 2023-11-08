import tensorflow as tf

def flatten_system_state(system_state):
    plate_rot = system_state['plate_rot']
    plate_vel = system_state['plate_vel']
    ball_pos_system = system_state['ball_pos']
    ball_vel_system = system_state['ball_vel']
    flattened_state = tf.concat([plate_rot, plate_vel, ball_pos_system, ball_vel_system], axis=-1)
    return flattened_state

def flatten_setpoint(setpoint):
    ball_pos_setpoint = setpoint['ball_pos']
    ball_vel_setpoint = setpoint['ball_vel']
    flattened_setpoint = tf.concat([ball_pos_setpoint, ball_vel_setpoint], axis=-1)
    return flattened_setpoint

def dictionarize_system_state(flattened_state):
    plate_rot = flattened_state[:,0:2]
    plate_vel = flattened_state[:,2:4]
    ball_pos_system = flattened_state[:,4:6]
    ball_vel_system = flattened_state[:,6:8]
    system_state = {
        'plate_rot': plate_rot,
        'plate_vel': plate_vel,
        'ball_pos': ball_pos_system,
        'ball_vel': ball_vel_system
    }
    return system_state

def dictionary_setpoint(flattened_setpoint):
    ball_pos_setpoint = flattened_setpoint[:,0:2]
    ball_vel_setpoint = flattened_setpoint[:,2:4]
    setpoint = {
        'ball_pos': ball_pos_setpoint,
        'ball_vel': ball_vel_setpoint
    }
    return setpoint

def get_ballstate(states, format):
    if isinstance(states, dict):
        states = flatten_system_state(states)
    assert states.shape[1] == 8, f"unflattened system state must have shape (batch, 8), received {states.shape}"

    if format == "dict":
        ball_state = {
            "ball_pos": states[:,4:6],
            "ball_vel": states[:,6:8]
        }
    elif format == "flat":
        ball_state = tf.concat([states[:,4:6], states[:,6:8]], axis=-1)
    else:
        raise ValueError(f"format must be either 'dict' or 'flat', received {format}")
    return ball_state

def get_setpoint(t, format):
    if isinstance(t, dict):
        if 'setpoint' in t:
            t = t['setpoint']
        else:
            assert 'ball_pos' in t and 'ball_vel' in t, f'expects dict with setpoint keys, or ball_pos and ball_vel keys, received {t.keys()}'
            assert 'plate_rot' not in t and 'plate_vel' not in t, f'input dictionary should not contain plate_rot or plate_vel keys, the input dictionary might not contain setpoint'
            t = {'ball_pos': t['ball_pos'], 'ball_vel': t['ball_vel']}

        if format == "dict":
            return t 
        elif format == "flat":
            return flatten_setpoint(t)

    # case t is flat
    raise NotImplementedError("Cannot get setpoint from flat tensor")        
