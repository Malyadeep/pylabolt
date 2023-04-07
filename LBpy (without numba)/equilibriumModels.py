def secondOrder(f_eq, u, rho, cs_2, cs_4, c, w):
    t2 = u[0] * u[0] + u[1] * u[1]
    for k in range(f_eq.shape[0]):
        t1 = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * rho * (1 + t1 * cs_2 +
                                0.5 * t1 * t1 * cs_4 -
                                0.5 * t2 * cs_2)


def firstOrder(f_eq, u, rho, cs_2, c, w):
    for k in range(f_eq.shape[0]):
        t1 = c[k, 0] * u[0] + c[k, 1] * u[1]
        f_eq[k] = w[k] * rho * (1 + t1 * cs_2)
