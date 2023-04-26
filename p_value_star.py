

def p_value_star(p_value):
    if 0.05 >= p_value > 0.01:
        p = '*'
    elif 0.01 >= p_value > 0.001:
        p = '**'
    elif p_value <=0.001:
        p = '***'
    else:
        p = 'ns'
    return p