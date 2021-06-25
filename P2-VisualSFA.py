import numpy as np 
from matplotlib import pyplot as plt
import mdp

series_length = 512
t = np.linspace(0,4*np.pi,series_length)
mean = 0
std = 1
a1 = np.random.normal(mean, std, size = series_length)
a2 = np.random.normal(mean, std, size = series_length)
a3 = np.random.normal(mean, std, size = series_length)
a4 = np.random.normal(mean, std, size = series_length)
a5 = np.random.normal(mean, std, size = series_length)

x1 = a1*np.sin(t+4*a3)
x2 = a2*np.sin(t+4*a4)
x3 = a2*np.sin(t+4*a4+np.pi/4)
x4 = a2*np.sin(t+4*a4+2*np.pi/4+0.5*a5)
x5 = a2*np.sin(t+4*a4+3*np.pi/4+0.5*a5)

plt.figure()
plt.subplot(2,3,1)
plt.plot(t,x1)
plt.subplot(2,3,2)
plt.plot(t,x2)
plt.subplot(2,3,3)
plt.plot(t,x3)
plt.subplot(2,3,4)
plt.plot(x1,x2)
plt.subplot(2,3,5)
plt.plot(x2,x3)

x1 = x1.reshape(series_length,1)
x2 = x2.reshape(series_length,1)
x3 = x3.reshape(series_length,1)
x4 = x4.reshape(series_length,1)
x5 = x5.reshape(series_length,1)

poly_deg = 3
time_emb = 10
dim = 1
flow = (mdp.nodes.EtaComputerNode() +
         mdp.nodes.TimeFramesNode(time_emb) +
         mdp.nodes.PolynomialExpansionNode(poly_deg) +
         mdp.nodes.SFANode(output_dim=dim) +
         mdp.nodes.EtaComputerNode())

flow.train(x1)
slow = flow(x1)
slow = slow.flatten()

print ('Eta value (time series): {}' .format(flow[0].get_eta(t=series_length)))
print ('Eta value (slow feature): {}' .format(flow[-1].get_eta(t=series_length-(time_emb-1))))

plt.figure()
plt.plot(np.arange(series_length-(time_emb-1)), slow)

plt.show()