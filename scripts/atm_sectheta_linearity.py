import am_model as am
import numpy as np
from matplotlib.pyplot import *

p=am.readamcfile('south_pole/SPole_winter.amc');
p.Nscale['h2o']=2000 # 2 mm pwv
m=am.am(prof=p);
m.df=20

za=np.arange(0,65,5)

I=[]
tau=[]
T=[]

for k,val in enumerate(za):
    m.za=val
    m.callam()
    m.parseresults()

    if k==0:
        I=m.I
        tau=m.tau
        T=m.Tb
    else:
        I=np.column_stack((I,m.I))
        tau=np.column_stack((tau,m.tau))
        T=np.column_stack((T,m.Tb))

f=m.f

# bands
# Top hat
bands=np.array([95,150,220])

#bw = 40.0 # tophat
bw = 'obs' # measured

# Actual bands
fn=['K95_frequency_spectrum_20150309.txt','B2_frequency_spectrum_20141216.txt',
    'K220_frequency_spectrum_20160120.txt']
win=[]
for k,val in enumerate(fn):
    if bw =='obs':
        bp=np.loadtxt(val,delimiter=',')
        x=bp[:,0];
        y=bp[:,1];
        bp=np.interp(f,x,y,left=0,right=0)
        if k==0:
            bp[np.bitwise_or(f<68,f>120)]=0;
        if k==1:
            bp[np.bitwise_or(f<110,f>190)]=0;
        if k==2:
            bp[np.bitwise_or(f<150,f>300)]=0;
    else:
        bp=np.zeros(f.shape)
        f0 = bands[k]-bw/2
        f1 = bands[k]+bw/2
        intind = np.where(np.bitwise_and(f>=f0,f<=f1))
        bp[intind]=1.0
        
    win.append(bp)
    

Iobs = np.zeros((za.size,bands.size))


for j,b in enumerate(win):

    for k,z in enumerate(za):
        Iobs[k,j]=(I[:,k]*b*m.df/1e3).sum()


amass = 1/np.cos(za*np.pi/180.0)

# Fit line to each
pp=[]
fit=[]
for k,val in enumerate(bands):
    pp.append(np.polyfit(amass,Iobs[:,k],1))
    fit.append(np.poly1d(pp[k]))

figure(1,figsize=(8,10))
clf()

subplot(3,1,1)
plot(f,I)
title('am code output for SPole winter {} mm PWV and za={}-{}'.format(p.Nscale['h2o']/1000.0,za[0],za[-1]))
ylabel('I (W/cm$^2$/GHz/sr)')
grid('on')

subplot(3,1,2)
plot(f,T)
ylabel('Tb (K)')
grid('on')

subplot(3,1,3)
semilogy(f,tau)
ylabel('tau')
grid('on')
xlabel('f (GHz)')
ylim([1e-3,1e2])
    
figure(2,figsize=(8,10))
clf()

subplot(2,1,1)
plot(amass,Iobs,'o')
legend(['95','150','220'],loc='upper left')
title('I integrated over {} GHz bandwidth'.format(bw))
ylabel('$\int I(f) df$ (W/cm$^2$/sr)')

gca().set_color_cycle(None)
for k,val in enumerate(bands):
    plot(amass,fit[k](amass))
grid('on')

subplot(2,1,2)
for k,val in enumerate(bands):
    resids = (Iobs[:,k]-fit[k](amass))/Iobs[:,k]
    plot(amass,resids,'o')
xlabel('airmass [sec($\\theta_z$)]')
title('fractional residuals with 1st order poly')
grid('on')
ylim([-.01,.01])


figure(1)
savefig('am_fig1.png')
figure(2)
savefig('am_fig2.png')
