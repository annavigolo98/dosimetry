

import numpy as np
import matplotlib.pyplot as plt
from scipy import  optimize
from scipy import e 


xfh,valfh=np.genfromtxt('fh.txt',dtype='float',comments='#',usecols=(0,1),unpack=True)  

print(len(xfh));
 
 
#plt.plot(xfh,valfh)

#plt.xscale('log')
#plt.yscale('log')

#plt.show()



dh=np.zeros(len(xfh),float)
dh[:]=xfh[:]*valfh[:]/sum(valfh*xfh)




yf=sum(valfh*xfh)

yd=sum(xfh*dh)

print("yF ",yf)
print("yd ",yd)


plt.xscale('log')
plt.plot(xfh,dh*xfh)
plt.show()    

#yF  4.5419097773169215
#yd  22.785969810525227


p_init=[0.,0.]
xplot=[]
yplot=[]

xfit=[]
yfit=[]

min1=107.
max1=185.

for i in range(len(xfh)):
   if(xfh[i]>min1 and xfh[i]<max1):
     xplot.append(xfh[i])
     yplot.append(dh[i]*xfh[i])

def fitfunc(x,b, c):
    return (0.00156*0.6)/(1+e**(b*(x-c)))

p_best, cov = optimize.curve_fit(
    fitfunc, xplot, yplot, p0=p_init          
) 

for j in range(0,1000):
	xfit.append(min1+((max1-min1)/1000*j))
	yfit.append(fitfunc(min1+((max1-min1)/1000*j),p_best[0],p_best[1])    )





perr = np.sqrt(np.diag(cov))

print("b =", p_best[0], "+-",perr[0])
print("c =", p_best[1], "+-",perr[1])



#parametri fit
#b = 0.07000947115751889 +- 0.0016898847238674976
#c = 139.8582248761132 +- 0.3544637324466947



plt.xscale('log')
plt.plot(xfh,dh*xfh)
plt.plot(xplot,yplot)
plt.plot(xfit,yfit)
plt.show()


#calibrazione->a=0.887
a=0.885

xfh[:]=xfh[:]*a
valfh[:]=a*valfh[:]

yf2=sum(valfh*xfh)
print('yf2 ',yf2)


dh2=np.zeros(len(xfh),float)
dh2=xfh[:]*valfh[:]/yf2

yd2=sum(dh2*xfh)
print('yd2 ',yd2)

#valori in keV/um
#yf2  3.5573372853390355
#yd2  20.16558328231509






plt.xscale('log')
plt.yscale('log')


plt.plot(xfh,valfh)
plt.plot(xfh,dh2)
plt.plot(xfh,dh2*xfh)
plt.show()



p_init2=[0.,1.]
xplot2=[]
yplot2=[]

xfit2=[]
yfit2=[]

min2=1.16
max2=1.7

for i in range(len(xfh)):
   if(xfh[i]>min2 and xfh[i]<max2):
     xplot2.append(xfh[i])
     yplot2.append(valfh[i])

def fitfunc2(x,m,k):
    return m*x**k
    

p_best2, cov2 = optimize.curve_fit(
    fitfunc2, xplot2, yplot2, p0=p_init2          
) 


print("m= ",p_best2[0])
print("k= ",p_best2[1])


for g in range(0,1000):
	xfit2.append(min2+(max2-min2)/1000*g)
	yfit2.append((fitfunc2(min2+((max2-min2)/1000*g),p_best2[0],p_best2[1])    ))



#plt.xscale('log')
#plt.yscale('log')

plt.plot(xfh,valfh)
plt.plot(xfit2,yfit2)
plt.plot(xplot2,yplot2)
plt.show()
 
