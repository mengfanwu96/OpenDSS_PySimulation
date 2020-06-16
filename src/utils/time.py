import re


duration_dict = {'w': 7*3600*24, 'd':3600*24, 'h':3600, 'm':60, 's':1}


def convert2seconds(time: str):
    t = re.split('([dhms])', time)
    assert '0' <= t[0] <= '9', "number not specified in %s." %time
    assert t[1] in duration_dict.keys(), "unit not found in %s." %time

    return float(t[0]) * duration_dict[t[1]]


def get_steps(time_span:str, step:str):
    duration = convert2seconds(time_span)
    step_size = convert2seconds(step)

    total_step = int(duration / step_size)
    assert total_step >= 1, "time span shorter than step size"

    return total_step


if __name__ == '__main__':
    print(get_steps('1d', '1m'))