import numpy as np 
from matplotlib import pyplot as plt
import mdp
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score

def normalize(X):
  return (X-np.mean(X))/np.std(X)

np.random.seed(42)
series_length = 512
t = np.linspace(0,4*np.pi,series_length)
mean = 0
std = 1
a1 = gaussian_filter(np.random.normal(mean, std, size = series_length), sigma=15)
a2 = gaussian_filter(np.random.normal(mean, std, size = series_length), sigma=15)
a3 = gaussian_filter(np.random.normal(mean, std, size = series_length), sigma=15)
a4 = gaussian_filter(np.random.normal(mean, std, size = series_length), sigma=15)
a5 = gaussian_filter(np.random.normal(mean, std, size = series_length), sigma=50)

x1 = (4+a1)*np.sin(t+4*a3)
x2 = (4+a2)*np.sin(t+4*a4)
x3 = (4+a2)*np.sin(t+4*a4+np.pi/4)
x4 = (4+a2)*np.sin(t+4*a4+2*np.pi/4+0.5*a5)
x5 = (4+a2)*np.sin(t+4*a4+3*np.pi/4+0.5*a5)
x = np.transpose(np.vstack((x1,x2,x3,x4,x5)))

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

plt.figure()
plt.subplot(1,3,1)
plt.plot(t,normalize(a1))
plt.subplot(1,3,2)
plt.plot(t,normalize(a2))
plt.subplot(1,3,3)
plt.plot(t,normalize(a5))

output_dim = 3
time_embd = 2
poly_exp = 2
flow = (
         mdp.nodes.EtaComputerNode() +
         mdp.nodes.PCANode(output_dim=output_dim) +
         mdp.nodes.TimeFramesNode(time_embd) +
         mdp.nodes.PolynomialExpansionNode(poly_exp) +
         mdp.nodes.SFANode(output_dim=output_dim) +
         mdp.nodes.EtaComputerNode()
         )

flow.train(x)
slow = flow(x)

plt.figure()
plt.subplot(1,3,1)
plt.plot(np.arange(series_length+1-time_embd)*4*np.pi/512, slow[:,0], \
    label = '{:4.2f}'.format(flow[-1].get_eta(t=series_length)[0]))
plt.legend()
plt.subplot(1,3,2)
plt.plot(np.arange(series_length+1-time_embd)*4*np.pi/512, slow[:,1], \
    label = '{:4.2f}'.format(flow[-1].get_eta(t=series_length)[1]))
plt.legend()
plt.subplot(1,3,3)
plt.plot(np.arange(series_length+1-time_embd)*4*np.pi/512, slow[:,2], \
    label = '{:4.2f}'.format(flow[-1].get_eta(t=series_length)[2]))
plt.legend()

print('r2: {:4.3}'.format(r2_score(slow[:,0], normalize(a5[:1-time_embd]))))
plt.show()