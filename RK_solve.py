# -*- coding: utf-8 -*-
from sys import stdin, stdout, setrecursionlimit
from gc import disable
from math import fabs, pi, sin, cos, hypot
from numpy import arange, array
from matplotlib import pyplot as plt

gets = input
puts = print
input = stdin.readline
print = stdout.write

N = 7
Xi, A, B = 1/14, 1/14, 1/15
EPS = 1e-4
"y1' =  Ay2"
"y2' = -By1"

def exact(x,x0,y0,z0):
    y = 210**0.5 * z0/14 * sin((x - x0)/210**0.5) + y0*cos((x - x0)/210**0.5)
    z = -210**0.5 * y0/15 * sin((x - x0)/210**0.5) + z0*cos((x - x0)/210**0.5)
    return y,z

def exacty(x):
    return (15/14)**0.5 / 14 * pi*sin(x/210**0.5) + pi/15*cos(x/210**0.5)

def exactz(x):
    return pi*(cos(x/210**0.5)/14 - (14/15)**0.5 / 15 * sin(x/210**0.5))

def f1(y2:float) -> float:
    return A*y2

def f2(y1:float) -> float:
    return -B*y1

def k11(y2:float) -> float:
    return f1(y2)

def k21(y1:float) -> float:
    return f2(y1)

def k12(h:float, y2:float, y1:float) -> float:
    return f1(y2 + h*Xi*k21(y1))

def k22(h:float, y1:float, y2:float) -> float:
    return f2(y1 + h*Xi*k11(y2))

def Runge_rule(f, xk:float, s, h:float) -> tuple:
    y1,y2 = f(xk, h)
    y3,y4 = f(xk, h/2)
    return (y3 - y1)/(2**s - 1), (y4 - y2)/(2**s - 1)

def Explicit_2nd_order_Runge_Kutta_method(xk:float, h:float = 0.10471975511965977) -> tuple:
    x, y1, y2 = 0, B*pi, A*pi
    y1h, y2h = y1, y2
    b1 = 1 - 1/(2*Xi)
    b2 = 1/(2*Xi)
    while (round(x,12) < round(xk,12)):
        y1h = y1 + h*(b1*k11(y2) + b2*k12(h,y2,y1))
        y2h = y2 + h*(b1*k21(y1) + b2*k22(h,y1,y2))
        y1,y2 = y1h, y2h
        x += h
    
    return y1, y2;

def Explicit_2nd_order_Runge_Kutta_method_auto(xk:float, rtol) -> tuple:
    
    x, y1, y2 = 0, B*pi, A*pi
    if (xk == 0): return (y1,y2,set([0]));
    y1h, y2h = y1, y2
    b1 = 1 - 1/(2*Xi)
    b2 = 1/(2*Xi)
    h = (EPS)**(1/3)*xk
    hy,hz = Runge_rule(Explicit_2nd_order_Runge_Kutta_method, xk, 2, h)
    h = h/2 * (rtol/hypot(hy,hz))**(1/2)    
    h = xk/(xk//h)
    h_next = h
    h_s = [h]
    x_s = [0]
    y1_s = [y1]
    y2_s = [y2]
    r_s = [0]    
    while (round(x,12) < round(xk,12)):
        h = xk - x if (xk - x < h) else h
        y1h = y1 + h*(b1*k11(y2) + b2*k12(h,y2,y1))
        y2h = y2 + h*(b1*k21(y1) + b2*k22(h,y1,y2))
        
        y1j_h = y1h
        y2j_h = y2h
        
        y1j_h2 = y1 + (h/2)*(b1*k11(y2) + b2*k12(h/2,y2,y1))
        y2j_h2 = y2 + (h/2)*(b1*k21(y1) + b2*k22(h/2,y1,y2))
        
        y1j_h2 += (h/2)*(b1*k11(y2j_h2) + b2*k12(h/2,y2j_h2,y1j_h2))
        y2j_h2 += (h/2)*(b1*k21(y1j_h2) + b2*k22(h/2,y1j_h2,y2j_h2))
        
        eyj_h,ezj_h = exact(x+h,x,y1,y2)
        eyj_h2,ezj_h2 = exact(x+h/2,x,y1,y2)
        eyj_h2,ezj_h2 = exact(x+h,x+h/2,eyj_h2,ezj_h2)
        l = hypot(fabs((eyj_h2 - eyj_h)/7), fabs((ezj_h2 - ezj_h)/7))        
        rho = hypot(fabs((y1j_h2 - y1j_h)/7), fabs((y2j_h2 - y2j_h)/7))
        tol = rtol*hypot(y1j_h2, y2j_h2) + 1e-12         
        if (rho > 4*tol):
            k = h
            while (rho > 4*tol):
                k /= 2
                y1h = y1 + k*(b1*k11(y2) + b2*k12(k,y2,y1))
                y2h = y2 + k*(b1*k21(y1) + b2*k22(k,y1,y2))
                
                y1j_h = y1h
                y2j_h = y2h
                
                y1j_h2 = y1 + (k/2)*(b1*k11(y2) + b2*k12(k/2,y2,y1))
                y2j_h2 = y2 + (k/2)*(b1*k21(y1) + b2*k22(k/2,y1,y2))
                
                y1j_h2 += (k/2)*(b1*k11(y2j_h2) + b2*k12(k/2,y2j_h2,y1j_h2))
                y2j_h2 += (k/2)*(b1*k21(y1j_h2) + b2*k22(k/2,y1j_h2,y2j_h2))                
                
            eyj_h,ezj_h = exact(x+k,x,y1,y2)
            eyj_h2,ezj_h2 = exact(x+k/2,x,y1,y2)
            eyj_h2,ezj_h2 = exact(x+k,x+k/2,eyj_h2,ezj_h2)
            l = hypot(fabs((eyj_h2 - eyj_h)/7), fabs((ezj_h2 - ezj_h)/7))
            r_s.append(l/rho)              
            y1,y2 = y1h, y2h
            x += k
            x_s.append(x)
            h_s.append(k)
            y1_s.append(y1)
            y2_s.append(y2)
            continue;
        
        elif (tol < rho <= 4*tol):
            h_next = h/2
        
        elif (tol <= 4*rho < 4*tol):
            h_next = h           
        elif (4*rho < tol):
            h_next = 1/7 if (1/7 < 2*h) else 2*h        
        r_s.append(l/rho)
        y1,y2 = y1h, y2h
        x += h
        x_s.append(x)
        h_s.append(h)
        y1_s.append(y1)
        y2_s.append(y2)        
        h = h_next
        
    return x_s, y1_s, y2_s, h_s, r_s;

K11 = lambda y2: A*y2;
K21 = lambda y1: -B*y1
K12 = lambda h,y2,y1: f1(y2 + h*(K21(y1)/2))
K22 = lambda h,y1,y2: f2(y1 + h*(K11(y2)/2))
K13 = lambda h,y2,y1: f1(y2 + h*((2**0.5 - 1)*K21(y1)/2 + (1 - 1/2**0.5)*K22(h,y1,y2)))
K23 = lambda h,y1,y2: f2(y1 + h*((2**0.5 - 1)*K11(y2)/2 + (1 - 1/2**0.5)*K12(h,y2,y1)))
K14 = lambda h,y2,y1: f1(y2 - h*(K22(h,y1,y2)/2**0.5 - (1 + 1/2**0.5)*K23(h,y1,y2)))
K24 = lambda h,y1,y2: f2(y1 - h*(K12(h,y2,y1)/2**0.5 - (1 + 1/2**0.5)*K13(h,y2,y1)))


def Gill_calculation_formula(xk, h):
    x, y1, y2 = 0, B*pi, A*pi
    y1h, y2h = y1, y2
    while (round(x,12) < round(xk,12)):
        y1h = y1 + h*(K11(y2)/6 + (1 - 1/2**0.5)*K12(h,y2,y1)/3 + (1 + 1/2**0.5)*K13(h,y2,y1)/3 + K14(h,y2,y1)/6)
        y2h = y2 + h*(K21(y1)/6 + (1 - 1/2**0.5)*K22(h,y1,y2)/3 + (1 + 1/2**0.5)*K23(h,y1,y2)/3 + K24(h,y1,y2)/6)
        y1,y2 = y1h, y2h
        x += h
    
    return y1, y2;       


def Gill_calculation_formula_auto(xk, rtol):
    x, y1, y2 = 0, B*pi, A*pi
    y1h, y2h = y1, y2
    
    h = EPS**(1/5)*xk
    hy,hz = Runge_rule(Gill_calculation_formula, xk, 4, h)
    h = h/2 * (rtol/hypot(hy,hz))**(1/4)
    h = xk/(xk//h)
    h_s = [h]
    x_s = [0]
    y1_s = [y1]
    y2_s = [y2]
    r_s = [0]
    while (round(x,12) < round(xk,12)):
        h = xk - x if (xk - x < h) else h
        y1h = y1 + h*(K11(y2)/6 + (1 - 1/2**0.5)*K12(h,y2,y1)/3 + (1 + 1/2**0.5)*K13(h,y2,y1)/3 + K14(h,y2,y1)/6)
        y2h = y2 + h*(K21(y1)/6 + (1 - 1/2**0.5)*K22(h,y1,y2)/3 + (1 + 1/2**0.5)*K23(h,y1,y2)/3 + K24(h,y1,y2)/6)
        
        
        y1j_h = y1h
        y2j_h = y2h
        
        y1j_h2 = y1 + (h/2)*(K11(y2)/6 + (1 - 1/2**0.5)*K12(h/2,y2,y1)/3 + (1 + 1/2**0.5)*K13(h/2,y2,y1)/3 + K14(h/2,y2,y1)/6)
        y2j_h2 = y2 + (h/2)*(K21(y1)/6 + (1 - 1/2**0.5)*K22(h/2,y1,y2)/3 + (1 + 1/2**0.5)*K23(h/2,y1,y2)/3 + K24(h/2,y1,y2)/6)
        
        y1j_h2 += (h/2)*(K11(y2j_h2)/6 + (1 - 1/2**0.5)*K12(h/2,y2j_h2,y1j_h2)/3 + (1 + 1/2**0.5)*K13(h/2,y2j_h2,y1j_h2)/3 + K14(h/2,y2j_h2,y1j_h2)/6)
        y2j_h2 += (h/2)*(K21(y1j_h2)/6 + (1 - 1/2**0.5)*K22(h/2,y1j_h2,y2j_h2)/3 + (1 + 1/2**0.5)*K23(h/2,y1j_h2,y2j_h2)/3 + K24(h/2,y1j_h2,y2j_h2)/6)
        
        rho = hypot(fabs((y1j_h2 - y1j_h)/15), fabs((y2j_h2 - y2j_h)/15))
        tol = rtol*hypot(y1j_h2, y2j_h2) + 1e-12
        
        if (rho > 16*tol):
            k = h
            while (rho > 16*tol):
                k /= 2
                y1h = y1 + k*(K11(y2)/6 + (1 - 1/2**0.5)*K12(k,y2,y1)/3 + (1 + 1/2**0.5)*K13(k,y2,y1)/3 + K14(k,y2,y1)/6)
                y2h = y2 + k*(K21(y1)/6 + (1 - 1/2**0.5)*K22(k,y1,y2)/3 + (1 + 1/2**0.5)*K23(k,y1,y2)/3 + K24(k,y1,y2)/6)
                y1j_h = y1h
                y2j_h = y2h
                
                y1j_h2 = y1 + (k/2)*(K11(y2)/6 + (1 - 1/2**0.5)*K12(k/2,y2,y1)/3 + (1 + 1/2**0.5)*K13(k/2,y2,y1)/3 + K14(k/2,y2,y1)/6)
                y2j_h2 = y2 + (k/2)*(K21(y1)/6 + (1 - 1/2**0.5)*K22(k/2,y1,y2)/3 + (1 + 1/2**0.5)*K23(k/2,y1,y2)/3 + K24(k/2,y1,y2)/6)
                
                y1j_h2 += (k/2)*(K11(y2j_h2)/6 + (1 - 1/2**0.5)*K12(k/2,y2j_h2,y1j_h2)/3 + (1 + 1/2**0.5)*K13(k/2,y2j_h2,y1j_h2)/3 + K14(k/2,y2j_h2,y1j_h2)/6)
                y2j_h2 += (k/2)*(K21(y1j_h2)/6 + (1 - 1/2**0.5)*K22(k/2,y1j_h2,y2j_h2)/3 + (1 + 1/2**0.5)*K23(k/2,y1j_h2,y2j_h2)/3 + K24(k/2,y1j_h2,y2j_h2)/6)
                
                rho = hypot(fabs((y1j_h2 - y1j_h)/15), fabs((y2j_h2 - y2j_h)/15))
                tol = rtol*hypot(y1j_h2, y2j_h2) + 1e-12
            eyj_h,ezj_h = exact(x+k,x,y1,y2)
            eyj_h2,ezj_h2 = exact(x+k/2,x,y1,y2)
            eyj_h2,ezj_h2 = exact(x+k,x+k/2,eyj_h2,ezj_h2)
            l = hypot(fabs((eyj_h2 - eyj_h)/15), fabs((ezj_h2 - ezj_h)/15))
            r_s.append(l/rho)              
            y1,y2 = y1h, y2h
            x += k
            x_s.append(x)
            h_s.append(k)
            y1_s.append(y1)
            y2_s.append(y2)                
            continue
        
        elif (tol < rho < 16*tol):
            h_next = h/2
        
        elif (tol <= 16*rho < 16*tol):
            h_next = h           
        elif (16*rho < tol):
            h_next = 1/15 if (1/15 < 2*h) else 2*h         
        
        eyj_h,ezj_h = exact(x+h,x,y1,y2)
        eyj_h2,ezj_h2 = exact(x+h/2,x,y1,y2)
        eyj_h2,ezj_h2 = exact(x+h,x+h/2,eyj_h2,ezj_h2)
        l = hypot(fabs((eyj_h2 - eyj_h)/15), fabs((ezj_h2 - ezj_h)/15))
        r_s.append(l/rho)        
        x += h
        x_s.append(x)
        y1,y2 = y1h, y2h
        h_s.append(h)
        y1_s.append(y1)
        y2_s.append(y2)
        h = h_next
    return x_s, y1_s, y2_s, h_s, r_s; 


def main() -> int:
    disable()
    print("I\n")
    
    y1, y2 = Explicit_2nd_order_Runge_Kutta_method(pi,0.10471975511965977)
    r1, r2 = Runge_rule(Explicit_2nd_order_Runge_Kutta_method, pi, 2, 0.10471975511965977)
    print("  1\n    y1(pi) = %.4f\n    y2(pi) = %.4f\n"%(y1,y2))
    print("  2\n    R1 = %e\n    R2 = %e\n"%(r1,r2))
    print("  Точные значения\n    y1(pi) = %.7f..\n    y2(pi) = %.7f..\n"%(exacty(pi),exactz(pi)))
    
    print("\nII\n")
    x, y1, y2, h, r = Explicit_2nd_order_Runge_Kutta_method_auto(pi,1e-5)
    print("  1\n    y1(pi) = %.7f\n    y2(pi) = %.7f\n"%(y1[-1],y2[-1]))
    print("  Изменения h: ")
    for v in reversed(sorted(h)):
        print("%f  "%v)
    
    print("\nIII\n")
    y1, y2 = Gill_calculation_formula(pi, 0.4487989505128276)
    print("  1-3\n    Обычный шаг\n      y1(pi) = %.4f\n      y2(pi) = %.4f\n"%(y1,y2))
    x,y1, y2, h, r = Gill_calculation_formula_auto(pi,1e-5)
    print("    Авто шаг\n      y1(pi) = %.7f\n      y2(pi) = %.7f\n"%(y1[-1],y2[-1]))
    print("    Изменения h: ")
    for v in reversed(sorted(h)):
        print(    "%f  "%v)
    plt.rcParams["figure.figsize"] = (12,7)
    X = arange(0,pi+0.1,pi/7)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("R(x,h)", fontsize=16)
    Y = [exacty, exactz]
    for i in range(2):
        axes[i].plot(X, array([fabs(Gill_calculation_formula(v,pi/7)[i] - Y[i](v)) for v in X]), c = 'b')
        axes[i].grid()
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("R%d"%(i+1))  
        axes[i].axhline(y = 0, color='k')
        axes[i].axvline(x = 0, color='k')
    
    X,Y1,Y2,H,R = Explicit_2nd_order_Runge_Kutta_method_auto(pi,1e-5)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Explicit_2nd_order_Runge_Kutta_method_auto\n h(x) and local error", fontsize=16)
    for i in range(2):
        axes[i].plot(X,R if (i) else H, c = 'b')
        axes[i].grid()
        
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("e" if (i) else "h") 
        axes[i].axhline(y = 0, color='k')
        axes[i].axvline(x = 0, color='k')
    
    
    X,Y1,Y2,H,R = Gill_calculation_formula_auto(pi,1e-5)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Gill_calculation_formula_auto\n h(x) and local error", fontsize=16)
    for i in range(2):
        axes[i].plot(X,R if (i) else H, c = 'b')
        axes[i].grid()
        
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("e" if (i) else "h") 
        axes[i].axhline(y = 0, color='k')
        axes[i].axvline(x = 0, color='k')          
    return 0;


if (__name__ == "__main__"): 
    main()    