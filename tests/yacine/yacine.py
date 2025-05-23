import numpy as np
from scipy.integrate import quad

def joint_density(s, v, s_next, v_next, mu, k, theta, sigma, rho, delta_t):
    a = k * theta
    Dv = 0.5 * np.log(1 - rho**2) + np.log(sigma) + np.log(v_next)
    cm1 = (7 * sigma**4 * (s_next - s)**6) / (11520 * (-1 + rho**2)**3 * v**5) \
        - (7 * rho * sigma**3 * (s_next - s)**5 * (v_next - v)) / (1920 * (-1 + rho**2)**3 * v**5) \
        + ((-193 + 298 * rho**2) * sigma**2 * (s_next - s)**4 * (v_next - v)**2) / (11520 * (-1 + rho**2)**3 * v**5) \
        + (rho * (193 - 228 * rho**2) * sigma * (s_next - s)**3 * (v_next - v)**3) / (2880 * (-1 + rho**2)**3 * v**5) \
        + ((745 - 2648 * rho**2 + 2008 * rho**4) * (s_next - s)**2 * (v_next - v)**4) / (11520 * (-1 + rho**2)**3 * v**5) \
        + ((-745 * rho + 1876 * rho**3 - 1152 * rho**5) * (s_next - s) * (v_next - v)**5) / (5760 * (-1 + rho**2)**3 * sigma * v**5) \
        + ((945 - 2090 * rho**2 + 1152 * rho**4) * (v_next - v)**6) / (11520 * (-1 + rho**2)**3 * sigma**2 * v**5) \
        - (sigma**2 * (s_next - s)**4 * (v_next - v)) / (64 * (-1 + rho**2)**2 * v**4) \
        + (rho * sigma * (s_next - s)**3 * (v_next - v)**2) / (16 * (-1 + rho**2)**2 * v**4) \
        + ((3 - 6 * rho**2) * (s_next - s)**2 * (v_next - v)**3) / (32 * (-1 + rho**2)**2 * v**4) \
        + (rho * (-3 + 4 * rho**2) * (s_next - s) * (v_next - v)**4) / (16 * (-1 + rho**2)**2 * sigma * v**4) \
        + ((7 - 8 * rho**2) * (v_next - v)**5) / (64 * (-1 + rho**2)**2 * sigma**2 * v**4) \
        + (sigma**2 * (s_next - s)**4) / (96 * (-1 + rho**2)**2 * v**3) - (rho * sigma * (s_next - s)**3 *(v_next - v)) / (24 * (-1 + rho**2)**2 * v**3) \
        + ((-7 + 10 * rho**2) * (s_next - s)**2 * (v_next - v)**2) / (48 * (-1 + rho**2)**2 * v**3) \
        + ((7 * rho - 8 * rho**3) * (s_next - s) * (v_next - v)**3) / (24 * (-1 + rho**2)**2 * sigma * v**3) \
        + ((-15 + 16 * rho**2) * (v_next - v)**4) / (96 * (-1 + rho**2)**2 * sigma**2 * v**3) \
        - ((s_next - s)**2 * (v_next - v)) / (4 * (-1 + rho**2) * v**2) \
        + (rho * (s_next - s) * (v_next - v)**2) / (2 * (-1 + rho**2) * sigma * v**2) - (v_next - v)**3 / (4 * (-1 + rho**2) * sigma**2 * v**2) \
        + (sigma**2 * (s_next - s)**2 - 2 * rho * sigma * (s_next - s) * (v_next - v) + (v_next - v)**2) / (2 * (-1 + rho**2) * sigma**2 * v)

    c0 = (7 * sigma**4 * (s_next - s)**4) / (1920 * (-1 + rho**2)**2 * v**4) \
        - (sigma * (30 * a * rho + sigma * (-30 * mu + 7 * rho * sigma)) * (s_next - s)**3 * (v_next - v)) / (480 * (-1 + rho**2)**2 * v**4) \
        + ((540 * a * rho**2 + sigma * (-540 * mu * rho + (-97 + 160 * rho**2) * sigma)) * (s_next - s)**2 * (v_next - v)**2) / (2880 * (-1 + rho**2)**2 * v**4) \
        + ((-270 * a * rho * (-1 + 2 * rho**2) + sigma * (270 * mu * (-1 + 2 * rho**2) \
        + rho * (97 - 118 * rho**2) * sigma)) * (s_next - s) * (v_next - v)**3) / (1440*(-1 + rho**2)**2*sigma*v**4) \
        + ((360 * a * (-4 + 5 * rho**2) + sigma * (-360 * mu * rho * (-3 + 4 * rho**2) \
        + (-215 + 236 * rho**2) * sigma)) * (v_next - v)**4) / (5760 * (-1 + rho**2)**2 * sigma**2 * v**4) \
        + (sigma * (a * rho - mu * sigma) * (s_next - s)**3) / (24*(-1 + rho**2)**2*v**3) \
        + ((-3*a*rho**2 + sigma*(3*mu*rho + sigma - rho**2*sigma))* (s_next - s)**2*(v_next - v))/(24*(-1 + rho**2)**2*v**3) \
        + ((a*rho*(-7 + 10*rho**2) + sigma*(mu*(7 - 10*rho**2) + 2*rho*(-1 + rho**2)*sigma))*(s_next - s)* (v_next - v)**2)/(24*(-1 + rho**2)**2*sigma*v**3) \
        + ((a*(8 - 9*rho**2) + sigma*(mu*rho*(-7 + 8*rho**2) + sigma - rho**2*sigma))*(v_next - v)**3)/ (24*(-1 + rho**2)**2*sigma**2*v**3) \
        + (sigma**2*(s_next - s)**2)/(24*(-1 + rho**2)*v**2) + ((12*a + sigma*(-12*mu*rho + sigma))*(v_next - v)**2)/(24*(-1 + rho**2)*sigma**2*v**2) \
        - ((s_next - s)*(2*a*rho - 2*mu*sigma - 2*k*rho*v + sigma*v))/ (2*sigma*v - 2*rho**2*sigma*v) \
        + ((v_next - v)*(2*a - 2*mu*rho*sigma - 2*k*v + rho*sigma*v))/(2*sigma**2*v - 2*rho**2*sigma**2*v) \
        + ((6*a*rho - 6*mu*sigma + rho*sigma**2)*(s_next - s)*(v_next - v))/ (12*sigma*v**2 - 12*rho**2*sigma*v**2)

    c1 = (sigma*(a*rho - mu*sigma)*(s_next - s))/(12*(-1 + rho**2)*v**2) \
        + ((s_next - s)**2*(60*a**2*(1 + 2*rho**2) + 180*mu**2*sigma**2 \
        + 2*sigma**4 - 2*rho**2*sigma**4 - 60*a*sigma*(6*mu*rho + sigma - rho**2*sigma) \
        - 60*k**2*v**2 + 60*k*rho*sigma*v**2 - 15*sigma**2*v**2))/(2880*(-1 + rho**2)**2*v**3) \
        + ((v_next - v)*(-12*a**2 - 12*mu**2*sigma**2 + 4*mu*rho*sigma**3 - 2*sigma**4 + 2*rho**2*sigma**4 + 4*a*sigma*(6*mu*rho + (3 - 4*rho**2)*sigma) + 12*k**2*v**2 - 12*k*rho*sigma*v**2 + 3*sigma**2*v**2))/(48*(-1 + rho**2)*sigma**2*v**2) \
        + (1/(2880*(-1 + rho**2)**2*sigma**2*v**3))* ((v_next - v)**2*(180*a**2*(-3 + 4*rho**2) + 60*mu**2*(-7 + 10*rho**2)*sigma**2 - 240*mu*rho*(-1 + rho**2)*sigma**3 - 94*sigma**4 + 190*rho**2*sigma**4 - 96*rho**4*sigma**4 + 60*a*sigma*(mu*(14*rho - 20*rho**3) \
        + (9 - 23*rho**2 + 14*rho**4)*sigma) + 60*k**2*v**2 - 120*k**2*rho**2*v**2 - 60*k*rho*sigma*v**2 + 120*k*rho**3*sigma*v**2 + 15*sigma**2*v**2 - 30*rho**2*sigma**2*v**2)) + (1/(24*(-1 + rho**2)*sigma**2*v))* (12*a**2 + 12*mu**2*sigma**2 + 2*sigma**4 - 2*rho**2*sigma**4 \
        + 12*mu*(2*k*rho - sigma)*sigma* v + 12*k**2*v**2 - 12*k*rho*sigma*v**2 + 3*sigma**2*v**2 - 12*a*(2*mu*rho*sigma + sigma**2 - rho**2*sigma**2 + 2*k*v - rho*sigma*v)) + (1/(1440*(-1 + rho**2)**2*sigma*v**3))*((s_next - s)*(v_next - v)*(-60*a**2*(rho + 2*rho**3) \
        - 180*mu**2*rho*sigma**2 + 120*mu*(-1 + rho**2)*sigma**3 + 180*a*rho*sigma* (2*mu*rho + sigma - rho**2*sigma) \
        + rho*(2*(-1 + rho**2)*sigma**4 + 60*k**2*v**2 - 60*k*rho*sigma*v**2 + 15*sigma**2*v**2)))

    c2 = -((60*a**2*(-2 + rho**2) - 60*mu**2*sigma**2 + 23*(-1 + rho**2)*sigma**4 + 120*a*sigma*(mu*rho + sigma - rho**2*sigma))/(720*(-1 + rho**2)*v**2))

    lnpX = - np.log(2 * delta_t * np.pi) - Dv + cm1/delta_t + c0 + delta_t * c1 + 0.5 * delta_t**2 * c2
    return np.exp(lnpX)

def marginal_density(s, v, s_next, mu, k, theta, sigma, rho, delta_t, v_max):
    a = k * theta
    int_Dv = v_max * (0.5 * np.log(1 - rho**2) + np.log(sigma) + np.log(v_max) - 1)
    int_cm1 = v_max * (7 * sigma**4 * (s_next - s)**6) / (11520 * (-1 + rho**2)**3 * v**5) \
            - (7/2 * rho * sigma**3 * (s - s_next)**5 * (2 * v - v_max) * v_max) / (1920 * (-1 + rho**2)**3 * v**5) \
            + (1/3 * (298 * rho**2 - 193) * sigma**2 * (s - s_next)**4 * v_max * (-3 * v * v_max + v_max**2 + 3 * v**2)) / (11520 * (-1 + rho**2)**3 * v**5) \
            + (1/4 * rho * (193 - 228 * rho**2) * sigma * (s - s_next)**3 * (v**4 - (v - v_max)**4)) / (2880 * (-1 + rho**2)**3 * v**5) \
            + (1/5 * (2008 * rho**4 - 2648 * rho**2 + 745) * (s - s_next)**2 * (v**5 - (v - v_max)**5)) / (11520 * (-1 + rho**2)**3 * v**5) \
            + (1/6 * (-1152 * rho**5 + 1876 * rho**3 - 745 * rho) * (s_next - s) * ((v - v_max)**6 - v**6)) / (5760 * (-1 + rho**2)**3 * sigma * v**5) \
            + (1/7 * (1152 * rho**4 - 2090 * rho**2 + 945) * (v**7 - (v - v_max)**7)) / (11520 * (-1 + rho**2)**3 * sigma**2 * v**5) \
            - (1/2 * sigma**2 * (s - s_next)**4 * v_max * (v_max - 2 * v)) / (64 * (-1 + rho**2)**2 * v**4) \
            + (1/3 * rho * sigma * (s_next - s)**3 * v_max * (-3 * v * v_max + v_max**2 + 3 * v**2)) / (16 * (-1 + rho**2)**2 * v**4) \
            + (1/4 * (3 - 6 * rho**2) * (s - s_next)**2 * ((v - v_max)**4 - v**4)) / (32 * (-1 + rho**2)**2 * v**4) \
            + (1/5 * rho * (4 * rho**2 - 3) * (s_next - s) * (v**5 - (v - v_max)**5)) / (16 * (-1 + rho**2)**2 * sigma * v**4) \
            + (1/6 * (8 * rho**2 - 7) * (v**6 - (v - v_max)**6)) / (64 * (-1 + rho**2)**2 * sigma**2 * v**4) \
            + v_max * (sigma**2 * (s_next - s)**4) / (96 * (-1 + rho**2)**2 * v**3) \
            - (1/2 * rho * sigma * (s - s_next)**3 * (2 * v - v_max) * v_max) / (24 * (-1 + rho**2)**2 * v**3) \
            + (1/3 * (10 * rho**2 - 7) * (s - s_next)**2 * v_max * (-3 * v * v_max + v_max**2 + 3 * v**2)) / (48 * (-1 + rho**2)**2 * v**3) \
            + (1/4 * (7 * rho - 8 * rho**3) * (s_next - s) * ((v - v_max)**4 - v**4)) / (24 * (-1 + rho**2)**2 * sigma * v**3) \
            + (1/5 * (16 * rho**2 - 15) * (v**5 - (v - v_max)**5)) / (96 * (-1 + rho**2)**2 * sigma**2 * v**3) \
            - (1/2 * (s - s_next)**2 * v_max * (v_max - 2 * v)) / (4 * (-1 + rho**2) * v**2) \
            + (1/3 * rho * (s_next - s) * v_max * (-3 * v * v_max + v_max**2 + 3 * v**2)) / (2 * (-1 + rho**2) * sigma * v**2) \
            - (1/4 * ((v - v_max)**4 - v**4)) / (4 * (-1 + rho**2) * sigma**2 * v**2) \
            + (1/3 * v_max * (-3 * v * (v_max + 2 * rho * sigma * (s - s_next)) + 3 * rho * sigma * (s - s_next) * v_max + v_max**2 + 3 * sigma**2 * (s - s_next)**2 + 3 * v**2)) / (2 * (-1 + rho**2) * sigma**2 * v)
    
    int_c0 = v_max * (7 * sigma**4 * (s_next - s)**4) / (1920 * (-1 + rho**2)**2 * v**4) \
            - (1/2 * sigma * (30 * a * rho + sigma * (7 * rho * sigma - 30 * mu)) * (s - s_next)**3 * (2 * v - v_max) * v_max) / (480 * (-1 + rho**2)**2 * v**4) \
            + (1/3 * (540 * a * rho**2 + sigma * ((160 * rho**2 - 97) * sigma - 540 * mu * rho)) * (s - s_next)**2 * v_max * (-3 * v * v_max + v_max**2 + 3 * v**2)) / (2880 * (-1 + rho**2)**2 * v**4) \
            + (1/4 * (sigma * (270 * mu * (2 * rho**2 - 1) + rho * (97 - 118 * rho**2) * sigma) - 270 * a * rho * (2 * rho**2 - 1)) * (s_next - s) * ((v - v_max)**4 - v**4)) / (1440*(-1 + rho**2)**2*sigma*v**4) \
            + (1/5 * (360 * a * (5 * rho**2 - 4) + sigma * ((236 * rho**2 - 215) * sigma - 360 * mu * rho * (4 * rho**2 - 3))) * (v**5 - (v - v_max)**5)) / (5760 * (-1 + rho**2)**2 * sigma**2 * v**4) \
            + v_max * (sigma * (a * rho - mu * sigma) * (s_next - s)**3) / (24*(-1 + rho**2)**2*v**3) \
            + (1/2 * (3 * a * rho**2 + sigma * ((rho**2 - 1) * sigma - 3 * mu * rho)) * (s - s_next)**2 * (2 * v - v_max) * v_max) / (24*(-1 + rho**2)**2*v**3) \
            + (1/3 * (a * rho * (10 * rho**2 - 7) + sigma * (mu * (7 - 10 * rho**2) + 2 * rho * (rho**2 - 1) * sigma)) * (s_next - s) * v_max * (-3 * v * v_max + v_max**2 + 3 * v**2)) / (24*(-1 + rho**2)**2*sigma*v**3) \
            + (1/4 * (a * (8 - 9 * rho**2) + sigma * (mu * (8 * rho**2 - 7) * rho + rho**2 * (-sigma) + sigma)) * ((v - v_max)**4 - v**4)) / (24*(-1 + rho**2)**2*sigma**2*v**3) \
            + v_max * (sigma**2*(s_next - s)**2) / (24*(-1 + rho**2)*v**2) \
            + (1/3 * (12 * a + sigma * (sigma - 12 * mu * rho)) * v_max * (-3 * v * v_max + v_max**2 + 3 * v**2)) / (24*(-1 + rho**2)*sigma**2*v**2) \
            - v_max * ((s_next - s)*(2*a*rho - 2*mu*sigma - 2*k*rho*v + sigma*v))/ (2*sigma*v - 2*rho**2*sigma*v) \
            + (1/2 * (2 * a - 2 * k * v + rho * sigma * (v - 2 * mu)) * v_max * (v_max - 2 * v)) / (2*sigma**2*v - 2*rho**2*sigma**2*v) \
            + (1/2 * (6 * a * rho + sigma * (rho * sigma - 6 * mu)) * (s - s_next) * (2 * v - v_max) * v_max) / (12*sigma*v**2 - 12*rho**2*sigma*v**2)

    int_c1 = v_max * (sigma*(a*rho - mu*sigma)*(s_next - s)) / (12*(-1 + rho**2)*v**2) \
            + v_max * ((s_next - s)**2*(60*a**2*(1 + 2*rho**2) + 180*mu**2*sigma**2 + 2*sigma**4 - 2*rho**2*sigma**4 - 60*a*sigma*(6*mu*rho + sigma - rho**2*sigma) - 60*k**2*v**2 + 60*k*rho*sigma*v**2 - 15*sigma**2*v**2)) / (2880*(-1 + rho**2)**2*v**3) \
            + (v_max * (0.5 * v_max - v) * (-12*a**2 - 12*mu**2*sigma**2 + 4*mu*rho*sigma**3 - 2*sigma**4 + 2*rho**2*sigma**4 + 4*a*sigma*(6*mu*rho + (3 - 4*rho**2)*sigma) + 12*k**2*v**2 - 12*k*rho*sigma*v**2 + 3*sigma**2*v**2)) / (48*(-1 + rho**2)*sigma**2*v**2) \
            + (1/(2880*(-1 + rho**2)**2*sigma**2*v**3)) * (1/3 * ((v_max - v)**3 + v**3)) * (180*a**2*(-3 + 4*rho**2) + 60*mu**2*(-7 + 10*rho**2)*sigma**2 - 240*mu*rho*(-1 + rho**2)*sigma**3 - 94*sigma**4 + 190*rho**2*sigma**4 - 96*rho**4*sigma**4 + 60*a*sigma*(mu*(14*rho - 20*rho**3) + (9 - 23*rho**2 + 14*rho**4)*sigma) + 60*k**2*v**2 - 120*k**2*rho**2*v**2 - 60*k*rho*sigma*v**2 + 120*k*rho**3*sigma*v**2 + 15*sigma**2*v**2 - 30*rho**2*sigma**2*v**2) \
            + v_max * (1/(24*(-1 + rho**2)*sigma**2*v)) * (12*a**2 + 12*mu**2*sigma**2 + 2*sigma**4 - 2*rho**2*sigma**4 + 12*mu*(2*k*rho - sigma)*sigma* v + 12*k**2*v**2 - 12*k*rho*sigma*v**2 + 3*sigma**2*v**2 - 12*a*(2*mu*rho*sigma + sigma**2 - rho**2*sigma**2 + 2*k*v - rho*sigma*v)) \
            + (1/(1440*(-1 + rho**2)**2*sigma*v**3)) * (s_next - s) * (0.5 * v_max - v) * (-60*a**2*(rho + 2*rho**3) - 180*mu**2*rho*sigma**2 + 120*mu*(-1 + rho**2)*sigma**3 + 180*a*rho*sigma* (2*mu*rho + sigma - rho**2*sigma) + rho*(2*(-1 + rho**2)*sigma**4 + 60*k**2*v**2 - 60*k*rho*sigma*v**2 + 15*sigma**2*v**2))

    int_c2 = - v_max * ((60*a**2*(-2 + rho**2) - 60*mu**2*sigma**2 + 23*(-1 + rho**2)*sigma**4 + 120*a*sigma*(mu*rho + sigma - rho**2*sigma))/(720*(-1 + rho**2)*v**2))

    lnpX = - v_max * np.log(2 * delta_t * np.pi) - int_Dv + int_cm1/delta_t + int_c0 + delta_t * int_c1 + 0.5 * delta_t**2 * int_c2
    return np.exp(lnpX)