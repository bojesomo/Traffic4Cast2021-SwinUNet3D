import numpy as np
"""Using symbolic algorithm to get the multiplication component matrix"""


def hstar(h):
    h_out = [h[0]]
    for h_ in h[1:]:
        h_ = '-' + h_
        h_out.append(h_)
    return h_out


def zmult(h1, h2):
    ha, hb = h1[:len(h1)//2], h1[len(h1)//2:]
    hc, hd = h2[:len(h1)//2], h2[len(h1)//2:]

    # (a, b) (c, d) = (ac – db*, a*d + cb)
    ac = [x_ + y_ for (x_, y_) in zip(ha, hc)]
    db = [x_ + y_ for (x_, y_) in zip(hstar(hd), hb)]

    da = [x_ + y_ for (x_, y_) in zip(hd, ha)]
    bc = [x_ + y_ for (x_, y_) in zip(hb, hstar(hc))]

    hm_a = [f"{x_} -{y_}" for (x_,y_) in zip(ac, db)]
    hm_a.extend([f"{x_} {y_}" for (x_,y_) in zip(da, bc)])
    return hm_a


def qmult(h1, h2):
    ha, hb = h1[:len(h1)//2], h1[len(h1)//2:]
    hc, hd = h2[:len(h1)//2], h2[len(h1)//2:]

    # (a, b) (c, d) = (ac – d*b, da + bc*)
    ac = zmult(ha, hc)
    db = zmult(hstar(hd), hb)

    da = zmult(hd, ha)
    bc = zmult(hb, hstar(hc))

    hm_a = [f"{x_} {' '.join([f'-{t}' for t in y_.split()])}" for (x_,y_) in zip(ac, db)]
    hm_a.extend([f"{x_} {y_}" for (x_,y_) in zip(da, bc)])
    return hm_a


def omult(h1, h2):
    ha, hb = h1[:len(h1)//2], h1[len(h1)//2:]
    hc, hd = h2[:len(h1)//2], h2[len(h1)//2:]

    # (a, b) (c, d) = (ac – d*b, da + bc*)
    ac = qmult(ha, hc)
    db = qmult(hstar(hd), hb)

    da = qmult(hd, ha)
    bc = qmult(hb, hstar(hc))

    hm_a = [f"{x_} {' '.join([f'-{t}' for t in y_.split()])}" for (x_,y_) in zip(ac, db)]
    hm_a.extend([f"{x_} {y_}" for (x_,y_) in zip(da, bc)])
    return hm_a


def smult(h1, h2):
    ha, hb = h1[:len(h1)//2], h1[len(h1)//2:]
    hc, hd = h2[:len(h1)//2], h2[len(h1)//2:]

    # (a, b) (c, d) = (ac – d*b, da + bc*)
    ac = omult(ha, hc)
    db = omult(hstar(hd), hb)

    da = omult(hd, ha)
    bc = omult(hb, hstar(hc))

    hm_a = [f"{x_} {' '.join([f'-{t}' for t in y_.split()])}" for (x_,y_) in zip(ac, db)]
    hm_a.extend([f"{x_} {y_}" for (x_,y_) in zip(da, bc)])
    return hm_a


# used recursion to cater for hypercomplex mukti
def hmult(h1, h2):
    assert len(h1) == len(h2)
    n = len(h1)
    ha, hb = h1[:len(h1) // 2], h1[len(h1) // 2:]
    hc, hd = h2[:len(h1) // 2], h2[len(h1) // 2:]
    if n > 2:  # do recursion
        ac = hmult(ha, hc)
        db = hmult(hstar(hd), hb)

        da = hmult(hd, ha)
        bc = hmult(hb, hstar(hc))
    else:  # end recursion
        ac = [x_ + y_ for (x_, y_) in zip(ha, hc)]
        db = [x_ + y_ for (x_, y_) in zip(hstar(hd), hb)]

        da = [x_ + y_ for (x_, y_) in zip(hd, ha)]
        bc = [x_ + y_ for (x_, y_) in zip(hb, hstar(hc))]

    hm_a = [f"{x_} {' '.join([f'-{t}' for t in y_.split()])}" for (x_, y_) in zip(ac, db)]
    hm_a.extend([f"{x_} {y_}" for (x_, y_) in zip(da, bc)])
    return hm_a


def hmat(h1, h2):
    # if len(h1) == 2:
    #     hm = zmult(h1, h2)
    # elif len(h1) == 4:
    #     hm = qmult(h1, h2)
    # elif len(h1) == 8:
    #     hm = omult(h1, h2)
    # elif len(h1) == 16:
    #     hm = smult(h1, h2)
    hm = hmult(h1, h2)

    m_out = []
    for h_ in hm:
        m_temp = []
        for x_ in h2:
            for h1_ in h_.split():
                if x_ in h1_:
                    h2_ = h1_.replace(x_, '')
                    c_ = h2_.count('-')
                    if c_ % 2:
                        h2_ = f"-{h2_.replace('-','')}"
                    else:
                        h2_ = f"{h2_.replace('-','')}"
                    m_temp.append(h2_)
        m_out.append(m_temp)
    return m_out


def cmat(h1, h2):
    # if len(h1) == 2:
    #     hm = zmult(h1, h2)
    # elif len(h1) == 4:
    #     hm = qmult(h1, h2)
    # elif len(h1) == 8:
    #     hm = omult(h1, h2)
    # elif len(h1) == 16:
    #     hm = smult(h1, h2)
    hm = hmult(h1, h2)

    m_out = []
    for h_ in hm:
        m_temp = []
        for x_ in h2:
            for h1_ in h_.split():
                if x_ in h1_:
                    h2_ = h1_.replace(x_, '')
                    c_ = h2_.count('-')
                    if c_ % 2:
                        h2_ = -int(f"{h2_.replace('-','').replace('w','')}")
                    else:
                        h2_ = int(f"{h2_.replace('-','').replace('w','')}")
                    m_temp.append(h2_)
        m_out.append(m_temp)
    return m_out


def get_comp_mat(num_components=8):
    h1 = [f'w{component}' for component in range(num_components)]
    h2 = [f'f{component}f' for component in range(num_components)]
    return np.array(cmat(h1, h2))


def get_hmat(num_components=8):
    h1 = [f'w{component}' for component in range(num_components)]
    h2 = [f'f{component}f' for component in range(num_components)]
    return hmat(h1, h2)
