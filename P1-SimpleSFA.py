import numpy as np 
from matplotlib import pyplot as plt
import mdp

series_length = 300
S = np.zeros((series_length, 1), 'd')
D = np.zeros((series_length, 1), 'd')

S[0] = 0.6
for t in range(1, series_length):
    D[t] = np.sin(np.pi/75. * t) - t/150.
    S[t] = (3.7+0.35*D[t]) * S[t-1] * (1 - S[t-1])

plt.figure()
plt.plot(np.arange(series_length),S)

timeframes = mdp.nodes.TimeFramesNode(4)
timeframed_S = timeframes.execute(S)
cubic_expand = mdp.nodes.PolynomialExpansionNode(3)
cubic_expanded_S = cubic_expand(timeframed_S)
sfa = mdp.nodes.SFANode(output_dim=1)
slow = sfa.execute(cubic_expanded_S)
slow = slow.flatten()
padded_slow = np.concatenate([[slow[0]], slow, [slow[296]], [slow[296]]])
rescaled_D = (D - np.mean(D, 0)) / np.std(D, 0)
plt.figure()
plt.plot(np.arange(series_length), rescaled_D, 'b', label='Normalized D')
plt.plot(np.arange(series_length), padded_slow, 'g--', lw = 3, label='SFA estimation of D')

plt.show()