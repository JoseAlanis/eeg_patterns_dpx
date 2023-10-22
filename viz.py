# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cm
from matplotlib import colormaps, ticker

import seaborn as sns

from mne.viz import plot_brain_colorbar
from mne.viz.evoked import _evoked_sensor_legend  # noqa

from stats import within_subject_cis
from mvpa import get_dfs

def plot_contrast_tvals(inst, times, mask, figsize=(6.5, 5),
                        mask_params=None, xlim=None, clim=None,
                        lab_colorbar=None):
    if clim is None:
        clim = [None, None]

    if mask_params is None:
        mask_params = dict(marker='o', markerfacecolor='k',
                           markeredgecolor='w',
                           linewidth=0, markersize=5)

    ttp = len(times)
    widths = [1.0 for _ in times] + [0.2]

    gs_kw = dict(width_ratios=widths,
                 height_ratios=[0.4, 0.4, 1.2],
                 wspace=0.10)
    mnames_top_cbar = [str(n) for n, t in enumerate(times)] + ['cbar']
    mnames_top = [str(n) for n, t in enumerate(times)] + ['X']
    mnames_botom = ['bottom'] * (ttp + 1)

    fig, axd = plt.subplot_mosaic([mnames_top_cbar,
                                   mnames_top,
                                   mnames_botom],
                                  gridspec_kw=gs_kw,
                                  empty_sentinel="X",
                                  figsize=figsize,
                                  layout='constrained')
    picks = inst.ch_names
    if len(inst.ch_names) == 64:
        picks = ['O2', 'Oz', 'Iz',
                 'P10', 'PO8', 'PO4', 'POz',
                 'P8', 'P6', 'P4', 'P2', 'Pz',
                 'TP8', 'T8',
                 'CP6', 'CP4', 'CP2', 'CPz',
                 'C6', 'C4', 'C2', 'Cz',
                 'FC6', 'FC4', 'FC2', 'FCz',
                 'FT8', 'F8','F6', 'F4', 'F2', 'Fz',
                 'AF8', 'AF4', 'AFz',
                 'Fp2', 'Fpz',
                 'Fp1',
                 'AF3', 'AF7',
                 'F1', 'F3', 'F5', 'F7', 'FT7',
                 'FC1', 'FC3', 'FC5',
                 'C1', 'C3', 'C5',
                 'CP1', 'CP3', 'CP5',
                 'T7', 'TP7',
                 'P1', 'P3', 'P5', 'P7', 'P9',
                 'PO3', 'PO7',
                 'O1'
                 ]
    inst.plot_image(picks=picks,
                    xlim=xlim,
                    clim=dict(eeg=clim),
                    colorbar=False,
                    mask=mask,
                    mask_cmap='RdBu_r',
                    mask_alpha=0.5,
                    show=False,
                    axes=axd['bottom'],
                    units=None,
                    scalings=dict(eeg=1),
                    )
    axd['bottom'].set_ylabel('Channels', labelpad=10.0, fontsize=11.0)
    axd['bottom'].set_xlabel('Time (ms)', labelpad=10.0, fontsize=11.0)
    axd['bottom'].tick_params(axis='x', labelrotation=-45)

    xmin = xlim[0]
    xmax = xlim[-1] + 0.01
    xticks = np.arange(xmin, xmax, 0.25)

    axd['bottom'].set_xticks(xticks, minor=False)
    axd['bottom'].set_xticklabels([int(tick * 1000) for tick in xticks])

    n_ch = len(inst.ch_names)
    yticks = np.arange(0, n_ch, 3)

    axd['bottom'].set_yticks(yticks, minor=False)
    labels = [picks[ch] for ch in yticks]
    axd['bottom'].set_yticklabels(labels, minor=False,
                                  fontdict={'fontsize': 8})

    if xlim is not None:
        axd['bottom'].spines['top'].set_visible(False)
        axd['bottom'].spines['right'].set_visible(False)
        axd['bottom'].spines['left'].set_bounds(-0.5, len(inst.ch_names) - 0.5)
        axd['bottom'].spines['bottom'].set_bounds(xlim[0], xlim[-1])

    axd['bottom'].axvline(x=0, ymin=0, ymax=len(inst.ch_names),
                          color='black', linestyle='dashed', linewidth=1.0)

    for text in axd['bottom'].texts:
        text.set_visible(False)
    axd['bottom'].set_title(None)

    plt.close('all')

    for nt, time in enumerate(times):
        inst.plot_topomap(times=[time],
                          mask=mask,
                          sphere='eeglab',
                          sensors=False,
                          mask_params=mask_params,
                          colorbar=False,
                          axes=axd[str(nt)],
                          scalings=dict(eeg=1),
                          vlim=tuple(clim),
                          time_unit='ms')
        con = ConnectionPatch(xyA=(0.0, -0.125),
                              coordsA=axd[str(nt)].transData,
                              xyB=(time, len(inst.ch_names)),
                              coordsB=axd['bottom'].transData,
                              color='k',
                              lw=2.0,
                              arrowstyle='->',
                              connectionstyle="arc3, rad=0.2")
        fig.add_artist(con)

    if lab_colorbar is None:
        lab_colorbar = r'$t$-value'
    plot_brain_colorbar(axd['cbar'],
                        transparent=False,
                        clim=dict(kind='value', lims=[clim[0], 0, clim[-1]]),
                        colormap='RdBu_r',
                        label=lab_colorbar)

    for label, ax in axd.items():
        # label physical distance in and down:
        trans = mtransforms.ScaledTranslation(
            -30 / 72, 30 / 72, fig.dpi_scale_trans
        )
        if label in ['0', 'bottom']:
            if label == 'bottom':
                lab = 'b |'
            else:
                lab = 'a |'
            ax.text(0.0, 1.0, lab, transform=ax.transAxes + trans,
                    fontsize='x-large', verticalalignment='top')

    return fig


def plot_contrast_sensor(inst, lower_b=None, upper_b=None, sig_mask=None,
                         sensors=['PO8', 'FCz'], xlim=[-0.25, 1.0],
                         ylim=[-15, 15], figsize=None, panel_letters=None,
                         legend_fontsize='medium',
                         scale=1, ylabel=None,
                         label='Cue B - Cue A'):
    # import letters
    from string import ascii_lowercase

    if figsize is None:
        figsize = (6.5, 3 * len(sensors))

    fig, ax = plt.subplots(nrows=len(sensors), ncols=1,
                           figsize=figsize)

    for si, sensor in enumerate(sensors):
        sens_ix = inst.ch_names.index(sensor)
        times = inst.times

        ax[si].plot(times, inst.get_data()[sens_ix] * scale,
                    label=label,
                    color='k')
        ax[si].set_ylim((ylim[0], ylim[1]))

        ax[si].invert_yaxis()
        ax[si].axhline(y=0, xmin=xlim[0], xmax=xlim[-1],
                       color='k', linestyle='dashed', linewidth=.75)
        ax[si].axvline(x=0, ymin=ylim[0], ymax=ylim[-1],
                       color='k', linestyle='dashed', linewidth=.75)
        ax[si].set_title('%s' % sensor)
        if sig_mask is not None:
            sig_times = [t for nt, t in enumerate(times) if
                         sig_mask[sens_ix, nt]]
            ax[si].bar(x=sig_times, height=ylim[1] - ylim[0], bottom=ylim[0],
                       width=1 / 128, alpha=0.15, color='gray')

        if (lower_b is not None) & (upper_b is not None):
            ax[si].fill_between(times,
                                lower_b[sens_ix] * scale,
                                upper_b[sens_ix] * scale,
                                color='lightgray',
                                edgecolor=None,
                                alpha=0.95)
        ax[si].legend(title="Contrast:", loc='lower right', alignment='left',
                      framealpha=0, fontsize=legend_fontsize)
        xticks = np.arange(xlim[0], xlim[-1] + 0.01, 0.25)
        ax[si].set_xticks(xticks)
        ax[si].set_xticklabels([int(tick * 1000) for tick in xticks])
        _evoked_sensor_legend(inst.info, picks=[sens_ix], ax=ax[si],
                              ymin=-5.0, ymax=-15.0,
                              show_sensors=True,
                              sphere='eeglab')
        ax[si].spines['top'].set_visible(False)
        ax[si].spines['right'].set_visible(False)
        ax[si].spines['left'].set_bounds(ylim[0], ylim[-1])
        ax[si].spines['bottom'].set_bounds(xlim[0], xlim[-1])
        ax[si].tick_params(axis='x', labelrotation=-45)
        ax[si].set_xlabel('Time (ms)')

        if ylabel is None:
            ax[si].set_ylabel(r'$t$-value')
        else:
            ax[si].set_ylabel(ylabel)

        trans = mtransforms.ScaledTranslation(
            -30 / 72, 30 / 72, fig.dpi_scale_trans
        )
        if panel_letters is None:
            pl = ascii_lowercase[si]
        else:
            pl = panel_letters[si]

        ax[si].text(0.0, 1.0, pl + ' |',
                    transform=ax[si].transAxes + trans,
                    fontsize='x-large', verticalalignment='top')

    plt.subplots_adjust(hspace=0.70, top=0.95, bottom=0.10)
    plt.close('all')

    return fig

def plot_erps_sensor(erps, times, info, cond_names=None,
                     sensors=['P5', 'Pz'], xlim=[-0.25, 1.0], ylim=[-7, 7],
                     figsize=None, panel_letters=None,
                     cmap=['magma', '0.15', '0.65'],
                     legend_loc='lower right',
                     legend_orient='vertical',
                     legend_size='medium',
                     label='Cue B - Cue A'):
    from string import ascii_lowercase

    channels = info.ch_names

    if cond_names is None:
        cond_names = [ascii_lowercase[i] for i in range(len(erps))]

    if figsize is None:
        figsize = (6.5, 3 * len(sensors))

    if legend_orient == 'horizontal':
        ncol = len(erps)
    else:
        ncol = 1

    ci = within_subject_cis(erps) * 1e6

    viridis = colormaps[cmap[0]]
    colors = np.linspace(float(cmap[1]), float(cmap[2]), len(erps))

    fig, ax = plt.subplots(nrows=len(sensors), ncols=1,
                           figsize=figsize)

    for nerp, erp in enumerate(erps):
        for si, sensor in enumerate(sensors):
            sens_ix = channels.index(sensor)

            mean_sig = erp[:, sens_ix, :].mean(0) * 1e6

            ax[si].plot(times, mean_sig,
                        label=cond_names[nerp],
                        color=viridis(colors[nerp]))
            ax[si].set_ylim((ylim[0], ylim[1]))

            ax[si].fill_between(times,
                                mean_sig - ci[nerp, sens_ix, :],
                                mean_sig + ci[nerp, sens_ix, :],
                                color=viridis(colors[nerp]),
                                edgecolor=None,
                                alpha=0.25)

            ax[si].invert_yaxis()
            ax[si].axhline(y=0, xmin=xlim[0], xmax=xlim[-1],
                           color='k', linestyle='dashed', linewidth=.75)
            ax[si].axvline(x=0, ymin=ylim[0], ymax=ylim[-1],
                           color='k', linestyle='dashed', linewidth=.75)
            ax[si].set_title('%s' % sensor)

            ax[si].legend(loc=legend_loc, alignment='left', framealpha=0,
                          ncol=ncol, fontsize=legend_size)
            xticks = np.arange(xlim[0], xlim[-1] + 0.01, 0.25)
            ax[si].set_xticks(xticks)
            ax[si].set_xticklabels([int(tick * 1000) for tick in xticks])

            _evoked_sensor_legend(info, picks=[sens_ix], ax=ax[si],
                                  ymin=-5.0, ymax=-15.0,
                                  show_sensors=True,
                                  sphere='eeglab')

            ax[si].spines['top'].set_visible(False)
            ax[si].spines['right'].set_visible(False)
            ax[si].spines['left'].set_bounds(ylim[0], ylim[-1])
            ax[si].spines['bottom'].set_bounds(xlim[0], xlim[-1])
            ax[si].tick_params(axis='x', labelrotation=-45)
            ax[si].set_ylabel(r'$\mu$V')
            ax[si].set_xlabel('Time (ms)')

            trans = mtransforms.ScaledTranslation(
                -30 / 72, 30 / 72, fig.dpi_scale_trans
            )
            if panel_letters is None:
                pl = ascii_lowercase[si]
            else:
                pl = panel_letters[si]

            ax[si].text(0.0, 1.0, pl + ' |',
                        transform=ax[si].transAxes + trans,
                        fontsize='x-large', verticalalignment='top')

    plt.subplots_adjust(hspace=0.75, top=0.95, bottom=0.10)
    plt.close('all')

    return fig


def plot_gat_matrix(data, times, stats_dict,
                    mask=None, vmax=None, vmin=None,
                    draw_mask=None, draw_contour=None,
                    draw_diag=True, draw_zerolines=True,
                    title_gat=None,
                    xlabel="Time (s)", ylabel="Time (s)",
                    figlabels=True, cmap="RdBu_r",
                    mask_alpha=.75, mask_cmap="RdBu_r",
                    test_times=None,
                    focus=None,
                    legend_loc='upper left',
                    comp_cmap='magma',
                    figsize=(8.0, 4.5)):
    """Return fig and ax for further styling of GAT matrix, e.g., titles
    """

    if test_times is None:
        test_times = {'P3': [3.0, 3.20]}

    comp_cmap = cm.get_cmap(comp_cmap)
    colors = np.linspace(0.15, 0.65, len(test_times.values()))

    gs_kw = dict(width_ratios=[0.25, 0.5, 0.25, 1.75],
                 height_ratios=[1.0, 0.75, 1.0, 0.75],
                 wspace=0.10)
    mnames_top = ['gat', 'gat', 'gat', 'diag']
    mnames_mid = ['gat', 'gat', 'gat', 'diag']
    mnames_bottom = ['gat', 'gat', 'gat', 'comp']
    mnames_cbar = ['X', 'cbar', 'X', 'comp']

    fig, axd = plt.subplot_mosaic([mnames_top,
                                   mnames_mid,
                                   mnames_bottom,
                                   mnames_cbar],
                                  gridspec_kw=gs_kw,
                                  empty_sentinel="X",
                                  figsize=figsize,
                                  layout='constrained')

    if vmax is None:
        vmax = np.abs(data).max()
    if vmin is None:
        vmax = np.abs(data).max()
        vmin = -vmax
    tmin, tmax = xlim = times[0], times[-1]
    extent = [tmin, tmax, tmin, tmax]
    im_args = dict(interpolation='nearest', origin='lower',
                   extent=extent, aspect='auto', vmin=vmin, vmax=vmax)

    if mask is not None:
        draw_mask = True if draw_mask is None else draw_mask
        draw_contour = True if draw_contour is None else draw_contour
    if any((draw_mask, draw_contour,)):
        if mask is None:
            raise ValueError("No mask to show!")

    if draw_mask:
        axd['gat'].imshow(data, alpha=mask_alpha, cmap=mask_cmap, **im_args)
        axd['gat'].imshow(np.ma.masked_where(~mask, data), cmap=cmap,
                          **im_args)
    else:
        axd['gat'].imshow(data, cmap=cmap, **im_args)
    if draw_contour and np.unique(mask).size == 2:
        big_mask = np.kron(mask, np.ones((10, 10)))
        axd['gat'].contour(big_mask, colors=["k"], extent=extent,
                           linewidths=[1],
                           corner_mask=False, antialiased=False, levels=[.5])
    axd['gat'].bar(0.01, 0.25, bottom=-0.25, width=0.5, alpha=0.75,
                   align='edge', color='k')
    axd['gat'].text(x=0.25,
                    y=-0.075,
                    s='cue',
                    fontdict=dict(fontsize=8, fontweight='bold', color='w'),
                    va='top',
                    ha='center')
    axd['gat'].bar(2.51, 0.25, bottom=-0.25, width=0.5, alpha=0.75,
                   align='edge', color='k')
    axd['gat'].text(x=2.75,
                    y=-0.075,
                    s='probe',
                    fontdict=dict(fontsize=8, fontweight='bold', color='w'),
                    va='top',
                    ha='center')
    axd['gat'].set_xlim(xlim)
    axd['gat'].set_ylim(xlim)

    if draw_diag:
        axd['gat'].plot((tmin, tmax), (tmin, tmax), color="k", linestyle=":",
                        linewidth=0.8)
    if draw_zerolines:
        axd['gat'].axhline(0, color="k", linestyle=":", linewidth=0.8)
        axd['gat'].axvline(0, color="k", linestyle=":", linewidth=0.8)

    axd['gat'].set_ylabel(ylabel + ', training', labelpad=5.0)
    axd['gat'].set_xlabel(xlabel + ', test', labelpad=5.0)

    axd['gat'].set_aspect(1. / axd['gat'].get_data_ratio())
    if title_gat is None:
        title_gat = "GAT Matrix"
    axd['gat'].set_title(title_gat, pad=10.0)

    axd['gat'].spines['top'].set_bounds(tmin, tmax)
    axd['gat'].spines['right'].set_bounds(tmin, tmax)
    axd['gat'].spines['left'].set_bounds(tmin, tmax)
    axd['gat'].spines['bottom'].set_bounds(tmin, tmax)
    axd['gat'].set_xticks(list(np.arange(0, tmax + 0.01, 0.5)), minor=False)
    axd['gat'].set_xticklabels([int(i * 1000) for i in list(np.arange(0, tmax + 0.01, 0.5))])
    axd['gat'].set_yticks(list(np.arange(0, tmax + 0.01, 0.5)), minor=False)
    axd['gat'].set_yticklabels([int(i * 1000) for i in list(np.arange(0, tmax + 0.01, 0.5))])
    axd['gat'].tick_params(axis='x', labelrotation=-45)

    if figlabels:
        trans = mtransforms.ScaledTranslation(
            -30 / 72, 30 / 72, fig.dpi_scale_trans
        )
        axd['gat'].text(0.0, 1.05, 'a |',
                        transform=axd['gat'].transAxes + trans,
                        fontsize='x-large', verticalalignment='top')

    plot_brain_colorbar(axd['cbar'],
                        transparent=False,
                        clim=dict(kind='value', lims=[vmin, .5, vmax]),
                        orientation='horizontal',
                        colormap=cmap,
                        label='Accuracy (%)')

    palette = [comp_cmap(colors[i]) for i, val in
               enumerate(test_times.values())]
    onsets = {k: v[0] for (k, v) in test_times.items()}
    axd['diag'].bar(onsets.values(), 1,
                    width=0.1, alpha=0.15, align='edge', color=palette)
    sns.lineplot(data=get_dfs(stats_dict, tmin=tmin, df_type='diag'),
                 color='k',
                 y='Accuracy (%)',
                 x='Time (s)',
                 errorbar=('ci', 99),
                 err_kws={'edgecolor': None},
                 ax=axd['diag'])
    axd['diag'].set_xlim((-0.25, 3.5))
    axd['diag'].set_xticks(np.arange(-0.25, 3.5 + 0.01, 0.5))

    if figlabels:
        axd['diag'].text(0.0, 1.0, 'b |',
                         transform=axd['diag'].transAxes + trans,
                         fontsize='x-large', verticalalignment='top')

    for t in stats_dict['diag'][2]:
        axd['diag'].scatter(t, 0.40, marker='|', color='k', s=25.0)

    axd['comp'].bar(onsets.values(), 1,
                    width=0.1, alpha=0.15, align='edge', color=palette)

    sns.lineplot(data=get_dfs(stats_dict, tmin=tmin, df_type=False),
                 hue='Component',
                 y='Accuracy (%)',
                 x='Time (s)',
                 errorbar=('ci', 99),
                 err_kws={'edgecolor': None},
                 palette=palette,
                 ax=axd['comp'])

    if focus is None:
        axd['comp'].set_xlim((-0.25, 3.5))
        axd['comp'].set_xticks(np.arange(-0.25, 3.5 + 0.01, 0.5))
        axd['comp'].set_xticklabels(np.arange(-0.25, 3.5 + 0.01, 0.5))
    elif focus == 'probe':
        axd['comp'].set_xlim((2.5, 3.5))
        axd['comp'].set_xticks(np.arange(2.5, 3.5 + 0.01, 0.25))
        axd['comp'].set_xticklabels(np.arange(2.5, 3.5 + 0.01, 0.25))
        axd['comp'].spines['bottom'].set_bounds(2.5, 3.5)
    elif focus == 'cue':
        axd['comp'].set_xlim((-0.25, 2.5))
        axd['comp'].set_xticks(np.arange(-0.25, 2.5 + 0.01, 0.5))
        axd['comp'].set_xticklabels(np.arange(-0.25, 2.5 + 0.01, 0.5))
        axd['comp'].spines['bottom'].set_bounds(-0.25, 2.5)

    if figlabels:
        axd['comp'].text(0.0, 1.0, 'c |',
                         transform=axd['comp'].transAxes + trans,
                         fontsize='x-large', verticalalignment='top')
    axd['comp'].legend(loc=legend_loc, alignment='left',
                       framealpha=0, fontsize='small')

    components = stats_dict.keys()
    components = [c for c in components if c != 'diag']
    max_off = (len(onsets) * 1.0) / 100
    offsets = np.linspace(0.45, 0.45 + max_off,
                          len(onsets)) - np.linspace(0.45,
                                                     0.45 + max_off,
                                                     len(onsets)).mean()
    for n_comp, comp in enumerate(components):
        for t in stats_dict[comp][2]:
            axd['comp'].scatter(t, 0.40 + offsets[n_comp], marker='|',
                                color=palette[n_comp], s=30.0)

    for ax in ['diag', 'comp']:
        if ax == 'diag':
            title = 'Diagonal decoding performance'
        else:
            title = 'Component generalisation across time'

        axd[ax].set_title(title, pad=10.0)
        axd[ax].set_ylim(0.40-0.025, 0.70 + 0.025)
        axd[ax].set_xticks(np.arange(0.0, 3.5 + 0.01, 0.5))
        axd[ax].set_xticklabels([int(i * 1000) for i in np.arange(0.0, 3.5 + 0.01, 0.5)])
        axd[ax].tick_params(axis='x', labelrotation=-45)

        axd[ax].set_yticks(np.arange(0.40, 0.75, 0.05))
        yt = [str(round(i, 2)).ljust(4, '0')
              for i in np.arange(0.40, 0.75, 0.05)]
        axd[ax].set_yticklabels(yt)
        axd[ax].set_xlabel('Time (ms)', labelpad=5.0)
        axd[ax].set_ylabel('Accuracy (%)', labelpad=5.0)
        axd[ax].spines['top'].set_visible(False)
        axd[ax].spines['right'].set_visible(False)
        axd[ax].spines['left'].set_bounds(0.40, 0.70)
        axd[ax].spines['bottom'].set_bounds(0, tmax)

        axd[ax].axhline(y=0.5, xmin=tmin, xmax=tmax,
                        color='black', linestyle='dashed', linewidth=.8)
        axd[ax].axvline(x=0.0, ymin=0, ymax=1.0,
                        color='black', linestyle='dashed', linewidth=.8)

    plt.close('all')

    return fig