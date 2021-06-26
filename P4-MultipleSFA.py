import numpy as np 
from matplotlib import pyplot as plt
import mdp
from scipy.ndimage import gaussian_filter

np.random.seed(42)
series_length = 512
t = np.linspace(0,4*np.pi,series_length)
mean = 0
std = 1
xs = gaussian_filter(np.random.normal(mean, std, size = series_length), sigma=10)
xf = gaussian_filter(np.random.normal(mean, std, size = series_length), sigma=3)

plt.figure()
plt.subplot(1,2,1)
plt.plot(t,xf)
plt.subplot(1,2,2)
plt.plot(t,xs)

x = np.transpose([xf, np.sin(2*xf)+0.5*xs])
flow = (
         mdp.nodes.EtaComputerNode() +
         mdp.nodes.PolynomialExpansionNode(2) +
         mdp.nodes.SFANode(output_dim=3) +
         mdp.nodes.PolynomialExpansionNode(2) +
         mdp.nodes.SFANode(output_dim=3) +
         mdp.nodes.PolynomialExpansionNode(2) +
         mdp.nodes.SFANode(output_dim=3) +
         mdp.nodes.PolynomialExpansionNode(2) +
         mdp.nodes.SFANode(output_dim=3) +
         mdp.nodes.EtaComputerNode()
         )

flow.train(x)
slow = flow(x)

plt.figure()
plt.subplot(1,3,1)
plt.plot(np.arange(series_length)*4*np.pi/512, slow[:,0], \
    label = '{:4.2f}'.format(flow[-1].get_eta(t=series_length)[0]))
plt.legend()
plt.subplot(1,3,2)
plt.plot(np.arange(series_length)*4*np.pi/512, slow[:,1], \
    label = '{:4.2f}'.format(flow[-1].get_eta(t=series_length)[1]))
plt.legend()
plt.subplot(1,3,3)
plt.plot(np.arange(series_length)*4*np.pi/512, slow[:,2], \
    label = '{:4.2f}'.format(flow[-1].get_eta(t=series_length)[2]))
plt.legend()

plt.show()