import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle

deter = pickle.load(open('cpi_10_det.out', "rb"))
stoch = pickle.load(open('cpi_10_sto.out', "rb"))
average = pickle.load(open('cpi_10_stoch.out', 'rb'))

df = pd.DataFrame({'CPI Deterministic': deter, 'CPI Stochatic': stoch, 'CPI Average': average})

matplotlib.style.use('ggplot')

# fig, ax = plt.subplots(2, 2, figsize=(10, 10),
#                        sharex=True, sharey=True)
# plt.plot()
ax = df.plot()
ax.set(xlabel='Iteration',
        ylabel='Discounted Reward',
        title='deterministic vs. stochastic policy')
# plt.show()
plt.savefig('cpi_policy_compare.png')

# for idx, method in enumerate(clusterScore):
#     df = pd.DataFrame(clusterScore[method])
#     df = df.transpose()
#     df.plot(ax=ax[idx / 2][idx % 2])
#     # lines, labels = ax[idx / 2][idx % 2].get_legend_handles_labels()
#     # ax[idx / 2][idx % 2].legend(lines, labels, loc='best')
#     ax[idx / 2][idx % 2].set(xlabel='Threshold',
#                              ylabel='Score',
#                              title=method)

# fig.tight_layout()
# fig.savefig('allInOneT.png')
