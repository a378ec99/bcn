from __future__ import division

import traceback
from functools import wraps
import subprocess
import json
from popen2 import popen2

import re

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import seaborn.apionly as sb




def clean(names):
    '''
    Clean up a list of file names or features by removing for example "_at" or ".CEL" at the end.
    '''
    cleaned  = [re.split(r'[_|.]', name)[0] for name in names]
    return cleaned

    
def skip_exceptions(func):
    '''
    Decorator to handle data based errors; returns 'exception' if some data based exception occurs.
    '''
    @wraps(func)
    def exception_handling(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            print traceback.format_exc()
            return None
    return exception_handling
    
    
def square_blocks_matrix(n, m_blocks, r=1.0, step=0):

    assert r >= 0
    assert n >= m_blocks

    blocksizes = np.repeat(n // m_blocks, m_blocks)
    blocksizes[-1] = blocksizes[-1] + (n % m_blocks)

    if step == 0:
        block_matrix = np.zeros((n, n))
    if step == 1:
        block_matrix = -1 * np.ones((n, n)) * r

    for i, size in enumerate(blocksizes):
        if step == 0:
            square = square_blocks_matrix(size, 2, r=r, step=1)
        if step == 1:
            square = np.ones((size, size)) * r
            di = np.diag_indices(size)
            square[di] = 1.0
        indices = np.indices(square.shape)
        indices = np.vstack([indices[0].ravel(), indices[1].ravel()]).T
        for index in indices:
            block_matrix[index[0] + (i * blocksizes[0]), index[1] + (i * blocksizes[0])] = square[index[0], index[1]]

    return block_matrix

  
def submit(kwargs, ppn=12, hours=10000, nodes=2, path='PUBLICATION/GIT/bcn'):
    
    mode = parameters['mode']
    class_ = parameters['class']

    if mode == 'local':
        subprocess.call(['python', 'taskpull_local.py', class_, json.dumps(kwargs)])
        
    if mode == 'parallel':
        output, input_ = popen2('qsub')
        job = """#!/bin/bash
                 #PBS -S /bin/bash
                 #PBS -l nodes={nodes}:ppn={ppn},walltime={hours}:00:00
                 #PBS -N  {jobname}
                 #PBS -o /home/sohse/projects/{path}/logs/{jobname}.out
                 #PBS -e /home/sohse/projects/{path}/logs/{jobname}.err
                 export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}"
                 export PATH=/home/sohse/anaconda2/bin:$PATH
                 echo $PATH
                 echo ""
                 echo -n "This script is running on: "
                 hostname -f
                 date
                 echo ""
                 echo "PBS_NODEFILE (${{PBS_NODEFILE}})"
                 echo ""
                 cat ${{PBS_NODEFILE}}
                 echo ""
                 cd $PBS_O_WORKDIR
                 /opt/openmpi/1.6.5/gcc/bin/mpirun python /home/sohse/projects/{path}/taskpull.py {class_} '{json}'
                 """.format(class_=class_, nodes=nodes, jobname=kwargs['name'], json=json.dumps(kwargs), ppn=ppn, hours=hours, path=path)
        input_.write(job)
        input_.close()
        print 'Submitted {output}'.format(output=output.read())
        # TODO monitor for completion and give signal that can continue!

    
def randomize_matrix(X):

    x_indices = np.arange(X.shape[0])
    y_indices = np.arange(X.shape[1])

    np.random.shuffle(x_indices)
    np.random.shuffle(y_indices)

    return X[x_indices[:, None], y_indices], x_indices, y_indices

   
class Visualize(object):

    def __init__(self, X=None, file_name='test', title=None, size=(8, 8)): # (22, 18)
        self.X = X
        if self.X is not None:
            self.mask = np.isnan(X)
        self.file_name = file_name
        self.title = title
        self.size = size
    """
    def depencence_structure(self, pairs, space='feature'):
        if space == 'feature':
            X = self.X
        if space == 'sample':
            X = self.X.T
        n = 10 # len(pairs)
        fig = pl.figure(figsize=(8, 8*n))
        for i, pair in enumerate(pairs[:10]):
            ax = fig.add_subplot(n, 1, i+1)
            ax.plot(X[pair[0]], X[pair[1]], '.', color='grey', alpha=0.6)
        fig.savefig(self.file_name)
    """
    def observed_matrix(self, vmin=-0.005, vmax=0.005, cmap=None, eps=False): # vmin=-0.01, vmax=0.01 # vmin=-3.0, vmax=3.0
        self.fig, self.ax = pl.subplots(figsize=self.size)
        if self.title:
            self.ax.set_title(self.title)
        if not cmap:
            cmap = pl.cm.viridis
        sb.heatmap(self.X, mask=self.mask, vmin=vmin, vmax=vmax, cmap=cmap, square=True, ax=self.ax, cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False)
        if eps:
            self.fig.savefig(self.file_name + '.eps', format='eps')
        self.fig.savefig(self.file_name + '.png')
        
    def recovery_performance(self, vmin=-2.5, vmax=2.5, xlabel=None, ylabel=None, xticklabels=None, yticklabels=None): # 5.0
        self.fig, self.ax = pl.subplots(figsize=self.size)
        if self.title:
            self.ax.set_title(self.title)
        else:
            self.ax.set_title('Bias recovery performance')
        cmap = pl.cm.inferno
        #temp = np.log2(self.X)
        #temp[temp > 1.0] = 1.0
        ax_seaborn = sb.heatmap(np.log2(self.X), mask=self.mask, vmin=vmin, vmax=vmax, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels, ax=self.ax, cbar_kws={'shrink': 0.5})
        cbar = ax_seaborn.collections[0].colorbar
        ax_seaborn.set_ylabel(ylabel)
        ax_seaborn.set_xlabel(xlabel)
        cbar.set_ticks([-2.0, -1.0, 0.0, 1.0, 2.0]) # vmin, vmax # np.log2([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf])
        cbar.set_ticklabels([r'$>4$x better', r'  $2$x better', '  Same', r'  $2$x worse', r'$>4$x worse']) # Perfect # , '4x worse', '8x worse (+)'
        self.ax.invert_yaxis()
        self.ax.tick_params(axis='both', which='both', length=0)
        self.fig.savefig(self.file_name)

    def dependence_structure(self, mixed, signal, pairs, stds, directions, estimate, guess, max_plots=10, max_points=10):

        n = min(max_plots, len(pairs))
        q = min(max_points, signal.shape[1])

        fig = pl.figure(figsize=(16, 16*n))

        for i in xrange(n):
            ax = fig.add_subplot(n, 1, i + 1)
            for j in xrange(q):
                ax.plot(signal[pairs[i][1]][j], signal[pairs[i][0]][j], 'o', color='green', alpha=0.6)
                ax.plot(mixed[pairs[i][1]][j], mixed[pairs[i][0]][j], 'o', color='red', alpha=0.6)
                ax.plot([signal[pairs[i][1]][j], mixed[pairs[i][1]][j]], [signal[pairs[i][0]][j], mixed[pairs[i][0]][j]], '-', color='red', alpha=0.6)
                ax.plot([signal[pairs[i][1]][j], estimate[pairs[i][1]][j]], [signal[pairs[i][0]][j], estimate[pairs[i][0]][j]], '-', color='blue', alpha=0.6)
                ax.plot(estimate[pairs[i][1]][j], estimate[pairs[i][0]][j], 'o', color='blue', alpha=0.6)
                ax.plot(guess[pairs[i][1]][j], guess[pairs[i][0]][j], 'o', color='brown', alpha=0.6)
                
            d = directions[i]
            b, a = stds[pairs[i]]
            b = b * -1 * d

            m = -b / float(a)
            ax.plot(list(ax.get_xlim()), [m * p + 0.0 for p in ax.get_xlim()], '-', color='orange', alpha=0.6)

            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_smart_bounds(True)
            ax.spines['bottom'].set_smart_bounds(True)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

            ax.yaxis.set_tick_params(width=4, which='major')
            ax.yaxis.set_tick_params(size=20, which='major')
            ax.xaxis.set_tick_params(width=4, which='major')
            ax.xaxis.set_tick_params(size=20, which='major')

            ax.yaxis.set_tick_params(width=4, which='minor')
            ax.yaxis.set_tick_params(size=10, which='minor')
            ax.xaxis.set_tick_params(width=4, which='minor')
            ax.xaxis.set_tick_params(size=20, which='minor')


            for axis in ['bottom', 'left']:
                ax.spines[axis].set_linewidth(4)

            ax.xaxis.set_major_formatter(pl.NullFormatter())
            ax.yaxis.set_major_formatter(pl.NullFormatter())

            #ax.set_xlim(ax.get_xlim())
            ax.axis('equal')

            #ax.tick_params(labelsize=20)

            #cax = plt.gcf().axes[-1]

            #cbar = fig.colorbar(cax, ticks=[-0.001, 0.001])
            #cbar.ax.set_yticklabels(['Low', 'High'])

            #cax.tick_params(labelsize=20)

        fig.savefig(self.file_name)

    def batch_effects(self, vmin=-2, vmax=4, xlabel=None, ylabel=None, x_loc=None, y_loc=None):
        self.fig, self.ax = pl.subplots(figsize=(16, 16))
        if self.title:
            self.ax.set_title(self.title)


        ax = self.ax
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.yaxis.set_tick_params(width=4, which='major')
        ax.yaxis.set_tick_params(size=20, which='major')
        ax.xaxis.set_tick_params(width=4, which='major')
        ax.xaxis.set_tick_params(size=20, which='major')

        ax.yaxis.set_tick_params(width=4, which='minor')
        ax.yaxis.set_tick_params(size=10, which='minor')
        ax.xaxis.set_tick_params(width=4, which='minor')
        ax.xaxis.set_tick_params(size=20, which='minor')


        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(4)

        ax.xaxis.set_major_formatter(pl.NullFormatter())
        ax.yaxis.set_major_formatter(pl.NullFormatter())

        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        
        #self.ax.set_xlabel(xlabel)
        #self.ax.set_ylabel(ylabel)
        cmap = pl.cm.inferno
        ax_seaborn = sb.heatmap(self.X, mask=self.mask, vmin=vmin, vmax=vmax, cmap=cmap, ax=self.ax, cbar_kws={'shrink': 0.5})
        cbar = ax_seaborn.collections[0].colorbar
        cbar.set_ticks([]) # vmin, vmax
        #cbar.set_ticklabels([])
        self.ax.invert_yaxis()
        self.ax.hlines(y_loc, 0, self.X.shape[1], linestyles='dashed', colors='white', linewidth=4.0)
        self.ax.vlines(x_loc, 0, self.X.shape[0], linestyles='dashed', colors='white', linewidth=4.0)

        self.ax.tick_params(axis='both', which='both', length=0)
        self.fig.savefig(self.file_name)
