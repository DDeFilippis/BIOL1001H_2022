""" 
David DeFilippis 2022-08-24
Procedural script for first two week of class on buttercup experiments.
"""


# dependency injection
import os
import numpy as np
import pandas as pd
import random as rdm
import statsmodels.api as sm
import tqdm
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl

mpl.rcParams['lines.markersize'] = 16
plt.rcParams.update({'font.size': 18})
# get the username
user = os.environ.get('USER')
path = f'/Users/{user}/Dropbox (Personal)/Marquette/BIOL1001H_2022/procedural_scripts/figures'
# Fixed bin count on histograms
hist_bins = 25
# plot the figure interactively
plt.ion()

# internal methods
def make_directory(input_path = f"{path}", verbose : bool = True):
	"""make sure that the directory we want to save to exists, if not make it"""
	import os # moving around the file system
	if not os.path.exists(input_path): # if the path doesn't exist
		os.makedirs(input_path) # lets make it
		if verbose:
			print(f"Directory {input_path} Created ")
	else:
		if verbose:
			print(f"Directory {input_path} already exists")

def save_figure(plt, base_path : str, filename : str, **kwargs):
	""" Saves figure with user specified path and file name """
	opt = {
		'dpi' : 300,
		'trans' : False,
		'verbose' : False
	}
	opt.update(kwargs)
	from datetime import datetime
	now = datetime.now()
	date_now = now.strftime("%Y-%m-%d") # get the current datetime
	make_directory(base_path, verbose=opt['verbose'])
	save_path = f"{base_path}/{date_now}_{filename}"
	plt.savefig(f"{save_path}", dpi=opt['dpi'], transparent=opt['trans'])


# load in the dataset. Bring in sheet 2 because people are terrible at properly storing data
bc_data = pd.read_excel(f'/Users/{user}/Dropbox (Personal)/Marquette/BIOL1001H_2022/data/buttercup_data_set.xlsx', sheet_name='Sheet2')


# first lets plot just the data as a histogram since we will be comparing means.
fig, ax = plt.subplots()
ax.hist(bc_data.loc[bc_data.location == 'top', 'plants'], fc="purple", bins=hist_bins, alpha=0.8, label='Top')
ax.hist(bc_data.loc[bc_data.location == 'bottom', 'plants'], fc="green", bins=hist_bins, alpha=0.8, label='Bottom')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Buttercups per Quadrat')
ax.set_ylabel('Frequency')
ax.set_xlim(0,12)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
save_figure(fig, path, '1_hist_of_raw_data')
ax.vlines(bc_data.loc[bc_data.location == 'top', 'plants'].mean(), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='purple', ls='-.', label='Mean')
ax.vlines(bc_data.loc[bc_data.location == 'bottom', 'plants'].mean(), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='green', ls='-.', label='Mean')
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
save_figure(fig, path, '2_hist_of_raw_data_with_means')


# lets get some different values at which we can draw samples from
vals = [1]*10 + [2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10] + [ int(x**1.5) for x in np.linspace(5, 63, 50) ]
vals = vals + [np.max(vals)]*10

size = len(vals)
# now we will use the data to extrapolate what the data would look like
# if the data were a normal distribution and we collected ~500 samples
samples = np.max(vals)
top_normal = np.random.normal(bc_data.loc[bc_data.location == 'top', 'plants'].mean(), bc_data.loc[bc_data.location == 'top', 'plants'].std(), samples)
bottom_normal = np.random.normal(bc_data.loc[bc_data.location == 'bottom', 'plants'].mean(), bc_data.loc[bc_data.location == 'bottom', 'plants'].std(), samples)
# plot the data
fig, ax = plt.subplots()
ax.hist(bc_data.loc[bc_data.location == 'top', 'plants'], fc="purple", bins=hist_bins, alpha=0.8, label='Top')
ax.hist(bc_data.loc[bc_data.location == 'bottom', 'plants'], fc="green", bins=hist_bins, alpha=0.8, label='Bottom')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(0, 12)
ax.set_ylim(0, 61.95)
ax.set_xlabel('Buttercups per Quadrat')
ax.set_ylabel('Frequency')
ax.vlines(top_normal.mean(), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='purple', ls='-.', label='Mean')
ax.vlines(bottom_normal.mean(), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='green', ls='-.', label='Mean')
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
save_figure(fig, path, '3_hist_of_raw_data_with_means_zoomed_out')

ax.hist(top_normal, fc="purple", bins=hist_bins, alpha=0.5)
ax.hist(bottom_normal, fc="green", bins=hist_bins, alpha=0.5)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
save_figure(fig, path, '4_hist_of_normal_dist')

# now we can look at the breadth of possible means we could get from the data.
top_boot = [ np.mean(rdm.choices(bc_data.loc[bc_data.location == 'top', 'plants'].values, k=len(bc_data.loc[bc_data.location == 'top', 'plants']))) for _ in range(samples) ]
bottom_boot = [ np.mean(rdm.choices(bc_data.loc[bc_data.location == 'bottom', 'plants'].values, k=len(bc_data.loc[bc_data.location == 'bottom', 'plants']))) for _ in range(samples) ]


def update(curr):
    """ draw our histogram with a specific chunk of the bootstrapped data """
    if curr == len(vals):
        ani.event_source.stop()
    plt.cla()
    ax.hist(bc_data.loc[bc_data.location == 'top', 'plants'], fc="purple", bins=hist_bins, alpha=0.8, label='Top')
    ax.hist(bc_data.loc[bc_data.location == 'bottom', 'plants'], fc="green", bins=hist_bins, alpha=0.8, label='Bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(0, 12)
    # get a normalized representation
    ax.hist(top_normal, fc="purple", bins=hist_bins, alpha=0.5)
    ax.hist(bottom_normal, fc="green", bins=hist_bins, alpha=0.5)
    # get the bootstrapped mean data
    ax.hist(top_boot[:curr], color='k', alpha=0.5)
    ax.hist(bottom_boot[:curr], color='k', alpha=0.5)
    # set the height of the graph
    # print(ax.get_ylim())
    if ax.get_ylim()[1] <= 61.95:
        ax.set_ylim(0, 61.95)
    # means from the normal data
    ax.vlines(top_normal.mean(), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='purple', ls='-.')
    ax.vlines(bottom_normal.mean(), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='green', ls='-.')
    # show where the mean and 95% of the bootstrapped data are
    ax.vlines(np.mean(top_boot[:curr]), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='k', ls='-.')
    ax.vlines(np.quantile(top_boot[:curr], 0.975), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='purple', ls=':')
    ax.vlines(np.quantile(top_boot[:curr], 0.025), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='purple', ls=':')

    ax.vlines(np.mean(bottom_boot[:curr]), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='k', ls='-.', label='Mean')
    ax.vlines(np.quantile(bottom_boot[:curr], 0.975), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='green', ls=':')
    ax.vlines(np.quantile(bottom_boot[:curr], 0.025), ax.get_ylim()[0], ax.get_ylim()[1]*.9, color='green', ls=':')
    ax.set_xlabel('Buttercups per Quadrat')
    ax.set_ylabel('Frequency')
    
    plt.gca().annotate('n={}'.format(curr), [plt.gca().get_xlim()[1]*.8, plt.gca().get_ylim()[1]*.9])
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()

# make an animation showing the mean and 95% of the data from the sample means with an increasing number of sample means. - this is to get us to think more deeply about the ideas behind what p-values and confidence/credible intervals mean.
fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update, vals, interval = 500, repeat=True, repeat_delay= 5000)
# plt.show() # comment out if you want to save the animation

f = f"{path}/{datetime.now().strftime('%Y-%m-%d')}_5_animated_bootstrapping.gif" 
gif_writer = animation.PillowWriter(fps=1.5) 
ani.save(f, writer=gif_writer)

# histograms and scatterplots of the other environmental variables
columns = ['lux', 'rh', 'temp', 'wind', 'pH', 'water']
xlabels = ['Light', 'Humidity', 'Temperature', 'Wind Speed', 'Soil pH', 'Soil Moisture']

for column, xlabel in zip(columns, xlabels):
    top_normal = np.random.normal(bc_data.loc[bc_data.location == 'top', column].mean(), bc_data.loc[bc_data.location == 'top', column].std(), 500)
    bottom_normal = np.random.normal(bc_data.loc[bc_data.location == 'bottom', column].mean(), bc_data.loc[bc_data.location == 'bottom', column].std(), 500)

    fig, ax = plt.subplots()
    # plot the actual data
    ax.hist(bc_data.loc[bc_data.location == 'top', column], fc="purple", bins=hist_bins, alpha=0.8, label='Top')
    ax.hist(bc_data.loc[bc_data.location == 'bottom', column], fc="green", bins=hist_bins, alpha=0.8, label='Bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, path, xlabel.lower().replace(' ', '_') + '_hist')
    # plot what we are assuming is normal data
    ax.hist(top_normal, fc="purple", bins=hist_bins, alpha=0.4)
    ax.hist(bottom_normal, fc="green", bins=hist_bins, alpha=0.4)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, path, xlabel.lower().replace(' ', '_') + '_hist_with_normal')


for column, xlabel in zip(columns, xlabels):
    fig, ax = plt.subplots()
    ax.scatter(bc_data.loc[bc_data.location == 'top', column], bc_data.loc[bc_data.location == 'top', 'plants'], ec='k', fc="purple", alpha=0.6, label='Top')
    ax.scatter(bc_data.loc[bc_data.location == 'bottom', column], bc_data.loc[bc_data.location == 'bottom', 'plants'], ec='k', fc="green", alpha=0.6, label='Bottom')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel("# Buttercups")
    ax.set_xlabel(xlabel)
    plt.legend()
    plt.tight_layout()
    save_figure(fig, path, xlabel.lower().replace(' ', '_') + '_scatter')

    x = sm.add_constant(bc_data[f'{column}'])
    y = bc_data.plants.tolist()

    # performing the regression
    # and fitting the model
    result = sm.OLS(y, x).fit()
    label = ''
    ax.plot(sorted(bc_data[f'{column}']), [y*result.params[f'{column}'] + result.params.const for y in sorted(bc_data[f'{column}'])], color='k', lw=2, alpha=0.8, label=label)
    plt.tight_layout()
    save_figure(fig, path, xlabel.lower().replace(' ', '_') + '_scatter_with_trendline')
    # printing the summary table
    print(column)
    print(result.summary())
 
# Can we make an animated example of a null permutation test of the difference between the means?

if curr == len(vals):
    ani.event_source.stop()
plt.cla()
ax.hist(bc_data.loc[bc_data.location == 'top', 'plants'], fc="purple", bins=hist_bins, alpha=0.8, label='Top')
ax.hist(bc_data.loc[bc_data.location == 'bottom', 'plants'], fc="green", bins=hist_bins, alpha=0.8, label='Bottom')

# make 1,000 shuffles of the locations
rdm_diffs = []
for i in tqdm.tqdm(range(np.max(vals))):
    bc_data[f'rdm_loc_{i}'] = rdm.sample(bc_data.location.tolist(), k=len(bc_data.location))
    rdm_diffs.append(bc_data.loc[bc_data[f'rdm_loc_{i}'] == 'top', 'plants'].mean() - bc_data.loc[bc_data[f'rdm_loc_{i}'] == 'bottom', 'plants'].mean())

bc_data[f'rdm_loc_{500}'] = rdm.sample(bc_data.location.tolist(), k=len(bc_data.location))
rdm_diffs.append(bc_data.loc[bc_data[f'rdm_loc_{i}'] == 'top', 'plants'].mean() - bc_data.loc[bc_data[f'rdm_loc_{500}'] == 'bottom', 'plants'].mean())
# for each reshuffle get the mean difference
mean_diff = bc_data.loc[bc_data.location == 'top', 'plants'].mean() - bc_data.loc[bc_data.location == 'bottom', 'plants'].mean()

# for each of the mean differences plot the accruing data and the one-tailed and two-tailed 95% lines along with the mean difference from the actual data
def draw_mean_diff(curr):
    """ draw our histogram with a specific chunk of the bootstrapped data """
    if curr == len(vals):
        ani.event_source.stop()
    axes[0].clear()
    # top plot shows the random data and the calculated mean difference
    axes[0].hist(bc_data.loc[bc_data[f'rdm_loc_{curr}'] == 'bottom', 'plants'], fc="green", alpha=0.65, label='Bottom')
    axes[0].hist(bc_data.loc[bc_data[f'rdm_loc_{curr}'] == 'top', 'plants'], fc="purple", alpha=0.65, label='Top')

    axes[0].set_xlim(3, 8)
    axes[0].set_ylim(0, 5)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].set_xlabel('Random Reassignment of Buttercups')
    # axes[0].set_ylabel('Frequency')
    axes[0].text(x=axes[0].get_xlim()[1]*.5, y=axes[0].get_ylim()[1]*.9, s=f'mean diff={round(rdm_diffs[curr], 4)}')
    
    axes[1].clear()
    # bottom plot shows the distribution of the differences in means from the randomized data and compares the 95% accumulation threshold with the mean from the actual data
    axes[1].hist(rdm_diffs[:curr], fc="green", alpha=0.8)
    axes[1].vlines(np.quantile(rdm_diffs[:curr], 0.95), axes[1].get_ylim()[0], axes[1].get_ylim()[1]*.9, color='green', ls='-.')
    axes[1].vlines(np.quantile(rdm_diffs[:curr], 0.99), axes[1].get_ylim()[0], axes[1].get_ylim()[1]*.9, color='green', ls=':')
    axes[1].vlines(mean_diff, axes[1].get_ylim()[0], axes[1].get_ylim()[1]*.9, color='purple', ls='-')
    axes[1].text(x=axes[1].get_xlim()[1]*.6, y=axes[1].get_ylim()[1]*.9, s=f'n={curr}')

    axes[1].set_xlim(-3, 3)
    # axes[1].set_ylim(0, 150)

    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].set_xlabel('Mean Difference in Buttercups')
    # axes[1].set_ylabel('Frequency')

    plt.tight_layout()

# make an animation showing the mean and 95% of the data from the sample means with an increasing number of sample means. - this is to get us to think more deeply about the ideas behind what p-values and confidence/credible intervals mean.
fig, axes = plt.subplots(nrows=2, ncols=1)
ani = animation.FuncAnimation(fig, draw_mean_diff, vals, interval = 500, repeat=True, repeat_delay= 5000)
# plt.show() # comment out if you want to save the animation

f = f"{path}/{datetime.now().strftime('%Y-%m-%d')}_6_perm_test_mean_diff.gif" 
gif_writer = animation.PillowWriter(fps=1.5) 
ani.save(f, writer=gif_writer)
