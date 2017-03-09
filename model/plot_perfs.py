#!/usr/bin/env python
# Show overview plot of models and performances

import matplotlib.pyplot as plt


NaN = float('NaN')
models = ['HMax', 'AlexNet', 'VGG16', 'ResNet']
datasets = ['Animal-2', 'Caltech-101', 'ILSVRC-1000']
colorspecs = ['ko-', 'k^-', 'kx-']
performances = dict()
years = [2007, 2012, 2014, 2015]
depths = [3, 8, 16, 156]
performances['Animal-2'] = [80, 97, NaN, NaN]
performances['Caltech-101'] = [42, 84.3, 92.7, NaN]
performances['ILSVRC-1000'] = [NaN, 62.5, 71.5, 77.0]

fig, ax1 = plt.subplots()
ax1.set_ylabel("Performance (%)")
ax1.set_xlabel('Year, Model')
fig.patch.set_alpha(0)
ax1.set_axis_bgcolor((1, 1, 1))
for colorspec,dataset in zip(colorspecs, datasets):
    ax1.plot(years, performances[dataset], colorspec, label=dataset)
ax1.plot([NaN, NaN], [NaN, NaN], 'ro-', label='Layer count')
plt.xticks(years, [str(y)+'\n'+m for y,m in zip(years, models)], rotation=0)
plt.yticks(range(0, 91, 20))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.36), ncol=4)
plt.ylim([0, 100])
ax2 = ax1.twinx()
ax2.plot(years, depths, 'ro-', label='Layer count')
ax2.set_ylabel('Layer count', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
fig.set_size_inches(10, 2.5, forward=True)
fig.subplots_adjust(top=0.82, bottom=0.25)
plt.margins(0.05)
fig.savefig('/home/sven2/python/misc/modelperfs.pdf')
plt.show()