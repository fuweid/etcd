#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

logging.basicConfig(format='[%(levelname)s %(asctime)s %(name)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

params = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='plot graph using mixed read/write result file.')
    parser.add_argument('input_file_a', type=str,
                        help='first input data files in csv format. (required)')
    parser.add_argument('input_file_b', type=str, nargs='?',
                        help='second input data files in csv format. (optional)')
    parser.add_argument('-t', '--title', dest='title', type=str, required=True,
                        help='plot graph title string')
    parser.add_argument('-o', '--output-image-file', dest='output', type=str, required=True,
                        help='output image filename')
    parser.add_argument('-F', '--output-format', dest='format', type=str, default='png',
                        help='output image file format. default: jpg')
    return parser.parse_args()


def load_data_files(*args):
    df_list = []
    try:
        for i in args:
            if i is not None:
                logger.debug('loading csv file {}'.format(i))
                df_list.append(pd.read_csv(i))
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    res = []
    try:
        for df in df_list:
            param_df = df[df['type'] == 'PARAM']
            param_str = ''
            if len(param_df) != 0:
                param_str = param_df['comment'].iloc[0]
            new_df = df[df['type'] == 'DATA'][[
                'ratio', 'conn_size', 'value_size']].copy()
            cols = [x for x in df.columns if x.find('iter') != -1]
            tmp = [df[df['type'] == 'DATA'][x].str.split(':') for x in cols]

            read_df = [x.apply(lambda x: float(x[0])) for x in tmp]
            read_avg = sum(read_df) / len(read_df)
            new_df['read'] = read_avg

            write_df = [x.apply(lambda x: float(x[1])) for x in tmp]
            write_avg = sum(write_df) / len(write_df)
            new_df['write'] = write_avg

            new_df['ratio'] = new_df['ratio'].astype(float)
            new_df['conn_size'] = new_df['conn_size'].astype(int)
            new_df['value_size'] = new_df['value_size'].astype(int)
            res.append({
                'dataframe': new_df,
                'param': param_str
            })
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    return res

# heatmap is copied from https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("white", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def human_sizes(size):
    """
    Returns human-readable string for a given bytes.
    """

    size = float(size)
    KiB = float(1024)
    MiB = float(KiB ** 2)
    if size < KiB:
        return '{0} Bytes'.format(size)
    elif KiB <= size < MiB:
        return '{0:.2f} KiB'.format(size / KiB)
    else:
        return '{0:.2f} MiB'.format(size / MiB)

# plot type is the type of the data to plot. Either 'read' or 'write'
def plot_data(title, plot_type, cmap_name_default, *args):
    if len(args) == 1:
        fig_size = (12, 16)
        df0 = args[0]['dataframe']
        df0param = args[0]['param']
        fig = plt.figure(figsize=fig_size)
        count = 0
        for val, df in df0.groupby('ratio'):
            count += 1
            plt.subplot(4, 2, count)

            connsize_labels = sorted(df['conn_size'].unique())
            # use reverse to ensure that it is ascending from bottom to top
            valuesize_labels = sorted(df['value_size'].unique())[::-1]
            # use reverse to ensure that it is ascending from bottom to top
            dataset = np.reshape(df[plot_type].to_numpy(), (-1, len(connsize_labels)))[::-1]

            valuesize_labels = [human_sizes(i) for i in valuesize_labels]

            im, _ = heatmap(dataset, valuesize_labels, connsize_labels,
                    cmap=cmap_name_default, cbarlabel="Requests/sec")
            annotate_heatmap(im, valfmt="{x:.1f}", size=7, fontweight="bold")

            plt.title('R/W Ratio {:.4f} [{:.2f}, {:.2f}]'.format(val, df[plot_type].min(),
                                                                 df[plot_type].max()))
            plt.ylabel("Value Size")
            plt.xlabel("Connections Amount")
            plt.tight_layout()
        fig.suptitle('{} [{}]\n{}'.format(title, plot_type.upper(), df0param))
    elif len(args) == 2:
        fig_size = (12, 26)
        df0 = args[0]['dataframe']
        df0param = args[0]['param']
        df1 = args[1]['dataframe']
        df1param = args[1]['param']
        fig = plt.figure(figsize=fig_size)
        col = 0
        delta_df = df1.copy()
        delta_df[[plot_type]] = ((df1[[plot_type]] - df0[[plot_type]]) /
                                 df0[[plot_type]]) * 100
        for tmp in [df0, df1, delta_df]:
            row = 0
            for val, df in tmp.groupby('ratio'):
                pos = row * 3 + col + 1
                plt.subplot(8, 3, pos)

                cmap_name = cmap_name_default
                cbarformat=None
                cbarlabel="Requests/sec"

                if col == 2:
                    cmap_name = 'hot'
                    cbarformat="%.2f%%"
                    cbarlabel=""

                connsize_labels = sorted(df['conn_size'].unique())
                # use reverse to ensure that it is ascending from bottom to top
                valuesize_labels = sorted(df['value_size'].unique())[::-1]
                # use reverse to ensure that it is ascending from bottom to top
                dataset = np.reshape(df[plot_type].to_numpy(), (-1, len(connsize_labels)))[::-1]

                valuesize_labels = [human_sizes(i) for i in valuesize_labels]

                im, cbar = heatmap(dataset, valuesize_labels, connsize_labels, 
                        cbar_kw={"format": cbarformat}, cbarlabel=cbarlabel, cmap=cmap_name)
                annotate_heatmap(im, valfmt="{x:.1f}", size=7, fontweight="bold")

                if row == 0:
                    if col == 0:
                        plt.title('{}\nR/W Ratio {:.4f} [{:.1f}, {:.1f}]'.format(
                            os.path.basename(params.input_file_a),
                            val, df[plot_type].min(), df[plot_type].max()))
                    elif col == 1:
                        plt.title('{}\nR/W Ratio {:.4f} [{:.1f}, {:.1f}]'.format(
                            os.path.basename(params.input_file_b),
                            val, df[plot_type].min(), df[plot_type].max()))
                    elif col == 2:
                        plt.title('Gain\nR/W Ratio {:.4f} [{:.2f}%, {:.2f}%]'.format(val, df[plot_type].min(),
                                                                                     df[plot_type].max()))
                else:
                    if col == 2:
                        plt.title('R/W Ratio {:.4f} [{:.2f}%, {:.2f}%]'.format(val, df[plot_type].min(),
                                                                               df[plot_type].max()))
                    else:
                        plt.title('R/W Ratio {:.4f} [{:.1f}, {:.1f}]'.format(val, df[plot_type].min(),
                                                                             df[plot_type].max()))
                plt.ylabel('Value Size')
                plt.xlabel('Connections Amount')
                plt.tight_layout()
                row += 1
            col += 1
        fig.suptitle('{} [{}]\n{}    {}\n{}    {}'.format(
            title, plot_type.upper(), os.path.basename(params.input_file_a), df0param,
            os.path.basename(params.input_file_b), df1param))
    else:
        raise Exception('invalid plot input data')
    fig.subplots_adjust(top=0.93)
    plt.savefig("{}_{}.{}".format(params.output, plot_type,
                params.format), format=params.format)


def main():
    global params
    logging.basicConfig()
    params = parse_args()
    result = load_data_files(params.input_file_a, params.input_file_b)
    for i in [('read', 'viridis'), ('write', 'plasma')]:
        plot_type, cmap_name = i
        plot_data(params.title, plot_type, cmap_name, *result)


if __name__ == '__main__':
    main()
