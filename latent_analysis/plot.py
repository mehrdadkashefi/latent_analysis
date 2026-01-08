"""
Collection of tools visualization
@Author: Mehrdad Kashefi
"""
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np


def RDM(x_dis, w=300, h=300, colormap = 'inferno', **kwargs):
    """Plot the RDM matrix

    Args:
        x_dis (np.array)
            Distance matrix
        w (int) 
            Width of the plot
        h (int)
            Height of the plot
        colormap (str)
            Colormap for the plot, default is 'inferno'
    **kwargs:
        cond_label (list)
            Condition labels
        cond_name (str)
            Condition name
        colorbar (bool)
            Show colorbar, default is False
        save_path (str)
            Path to save the plot, default is None
    """

    cond_label = kwargs.get('cond_label', list(range(x_dis.shape[0])))
    cond_name = kwargs.get('cond_name', '')
    colorbar = kwargs.get('colorbar', False)
    save_path = kwargs.get('save_path', None)

    fig = px.imshow(
        x_dis,
        labels=dict(x=cond_name, y=cond_name, color="Distance"),
        color_continuous_scale=colormap,  # You can choose other scales like 'Plasma', 'Cividis', etc.
        x=cond_label,
        y=cond_label,
    )

    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        autosize=True,
    )

    fig.update_layout(
        template="simple_white", 
        coloraxis_showscale=colorbar,
        width=w,height=h, 
        margin={'b':0, 'l':0, 't':0, 'r':0},  
        showlegend=False

    )
    if save_path is not None:
        fig.write_image(save_path)
        
    fig.show()

def line_1d(data, **kwargs):
    """ Plot a 1D line (like EMG or speed profile)

    Args:
        data (np.array)
            Data to plot (cond, time)
    **kwargs:
        t_range (list)
            Time range for x-axis
        title (str)
            Title of the plot, set '' for no title
        x_label (str)
            Label for x-axis
        y_label (str)
            Label for y-axis
        template (str)
            Plotly template for figure theme
        width (int)
            Width of the figure
        height (int)        
            Height of the figure
        legend_show (bool)
            Whether to show legend
        show (bool)
            Whether to show the figure
        alpha (float)
            Line transparency
        colors (list/str)
            List of colors for each condition, None for monochrome, 'default' for colorblind palette, or a list of colors as hex
    """
    t_range = kwargs.get('t_range', None)
    title = kwargs.get('title', '1D Line Plot')
    x_label = kwargs.get('x_label', 'Time')
    y_label = kwargs.get('y_label', '')
    template = kwargs.get('template', 'simple_white')
    width = kwargs.get('width', 800)
    height = kwargs.get('height', 500)
    legend_show = kwargs.get('legend_show', False)
    show = kwargs.get('show', True)
    alpha = kwargs.get('alpha', 1.0)
    colors = kwargs.get('colors', None)
    save_path = kwargs.get('save_path', None)
    
    
    if t_range is not None:
        tt = np.linspace(t_range[0], t_range[1], data.shape[1])
    else:
        tt = np.arange(data.shape[1])

    if colors is None:
        color_list = ['black' for _ in range(data.shape[0])]
    elif colors == 'default':
        color_list = sns.color_palette("colorblind", n_colors=data.shape[0]).as_hex()
    else:
        color_list = colors    
    
    fig = go.Figure()
    for cond in range(data.shape[0]):
        fig.add_trace(go.Scatter(x=tt, y=data[cond, :], mode='lines', opacity=alpha, line=dict(color=color_list[cond])))

    fig.update_layout(title=title,
                      xaxis_title=x_label,
                      yaxis_title=y_label)

    # change figure size and theme
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        template=template
    )
    
    # remove legends
    if not legend_show:
        fig.update_layout(showlegend=False)

    # save figure
    if save_path is not None:
        fig.write_image(save_path)
    
    if show:
        fig.show()

def line_2d(data, **kwargs):
    """ Plot a 2D line (like PCs or XY position)

    Args:
        data (np.array)
            Data to plot (cond, time)
    **kwargs:
        scatters_dict (dict)
            Scatter plot parameters
        title (str)
            Title of the plot, set '' for no title
        x_label (str)
            Label for x-axis
        y_label (str)
            Label for y-axis
        template (str)
            Plotly template for figure theme
        width (int)
            Width of the figure
        height (int)        
            Height of the figure
        force_equal_axis (bool)
            Whether to force the axis to be equal, default is False
        legend_show (bool)
            Whether to show legend
        show (bool)
            Whether to show the figure
        alpha (float)
            Line transparency
        colors (list/str)
            List of colors for each condition, None for monochrome, 'default' for colorblind palette, or a list of colors as hex
    """
 
    title = kwargs.get('title', '2D Line Plot')
    x_label = kwargs.get('x_label', 'Time')
    y_label = kwargs.get('y_label', '')
    template = kwargs.get('template', 'simple_white')
    width = kwargs.get('width', 800)
    height = kwargs.get('height', 500)
    force_equal_axis = kwargs.get('force_equal_axis', False)
    legend_show = kwargs.get('legend_show', False)
    show = kwargs.get('show', True)
    alpha = kwargs.get('alpha', 1.0)
    colors = kwargs.get('colors', None)
    save_path = kwargs.get('save_path', None)
    scatters_dict = kwargs.get('scatters_dict', None)

    num_dim = data.shape[-1]
    assert num_dim == 2 or num_dim == 3

 
    if colors is None:
        color_list = ['black' for _ in range(data.shape[0])]
    elif colors == 'default':
        color_list = sns.color_palette("colorblind", n_colors=data.shape[0]).as_hex()
    else:
        color_list = colors    


    if scatters_dict is not None:
        scatter_info_list = []
        for s_dict in scatters_dict:
            scatters_info = {}
            # time 
            scatters_info['time'] = s_dict['time']
            # size
            if type(s_dict['size']) == int:
                scatters_info['size'] = [s_dict['size'] for _ in range(data.shape[0])]
            else:
                scatters_info['size'] = s_dict['size']
            # color
            if s_dict['color'] == '':
                scatters_info['color'] = ['black' for _ in range(data.shape[0])]
            elif s_dict['color'] == 'same':       
                scatters_info['color'] = color_list
            else:
                scatters_info['color'] = s_dict['color']
            # opacity
            if type(s_dict['alpha']) == float:
                scatters_info['alpha'] = [s_dict['alpha'] for _ in range(data.shape[0])]
            else: 
                scatters_info['alpha'] = s_dict['alpha']
            # shape
            if type(s_dict['shape']) == str:
                scatters_info['shape'] = [s_dict['shape'] for _ in range(data.shape[0])]
            else:
                scatters_info['shape'] = s_dict['shape']

            scatter_info_list.append(scatters_info)
            

    
    fig = go.Figure()
    for cond in range(data.shape[0]):
        fig.add_trace(go.Scatter(x=data[cond, :, 0], y=data[cond, :, 1], mode='lines', opacity=alpha, line=dict(color=color_list[cond])))

    # add scatter points if provided
    if scatters_dict is not None:
        for scatters_info in scatter_info_list:
            for cond in range(data.shape[0]):
                fig.add_trace(go.Scatter(
                    x=[data[cond, scatters_info['time'], 0]],
                    y=[data[cond, scatters_info['time'], 1]],
                    mode='markers',
                    marker=dict(
                        size=scatters_info['size'][cond],
                        color=scatters_info['color'][cond],
                        opacity=scatters_info['alpha'][cond],
                        symbol=scatters_info['shape'][cond]
                    )
                ))
   
    fig.update_layout(title=title,
                      xaxis_title=x_label,
                      yaxis_title=y_label)

    # change figure size and theme
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        template=template
    )

    # force axis to be equal
    if force_equal_axis:
        fig.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1
            )
        )
    
    # remove legends
    if not legend_show:
        fig.update_layout(showlegend=False)

    # save figure
    if save_path is not None:
        fig.write_image(save_path)
    
    if show:
        fig.show()
 
def line_3d(data, **kwargs):
    """ Plot a 2D line (like PCs or XY position)

    Args:
        data (np.array)
            Data to plot (cond, time)
    **kwargs:
        scatters_dict (dict)
            Scatter plot parameters
        title (str)
            Title of the plot, set '' for no title
        x_label (str)
            Label for x-axis
        y_label (str)
            Label for y-axis
        z_label (str)
            Label for z-axis
        force_equal_axis (bool)
            Whether to force the axis to be equal, default is False
        template (str)
            Plotly template for figure theme
        width (int)
            Width of the figure
        height (int)        
            Height of the figure
        legend_show (bool)
            Whether to show legend
        show (bool)
            Whether to show the figure
        alpha (float)
            Line transparency
        colors (list/str)
            List of colors for each condition, None for monochrome, 'default' for colorblind palette, or a list of colors as hex
    """
 
    title = kwargs.get('title', '1D Line Plot')
    x_label = kwargs.get('x_label', 'Time')
    y_label = kwargs.get('y_label', '')
    z_label = kwargs.get('z_label', '')
    template = kwargs.get('template', 'simple_white')
    width = kwargs.get('width', 800)
    height = kwargs.get('height', 500)
    force_equal_axis = kwargs.get('force_equal_axis', False)
    legend_show = kwargs.get('legend_show', False)
    show = kwargs.get('show', True)
    alpha = kwargs.get('alpha', 1.0)
    colors = kwargs.get('colors', None)
    save_path = kwargs.get('save_path', None)
    scatters_dict = kwargs.get('scatters_dict', None)

    num_dim = data.shape[-1]
    assert num_dim == 2 or num_dim == 3

 
    if colors is None:
        color_list = ['black' for _ in range(data.shape[0])]
    elif colors == 'default':
        color_list = sns.color_palette("colorblind", n_colors=data.shape[0]).as_hex()
    else:
        color_list = colors    


    if scatters_dict is not None:
        scatter_info_list = []
        for s_dict in scatters_dict:
            scatters_info = {}
            # time 
            scatters_info['time'] = s_dict['time']
            # size
            if type(s_dict['size']) == int:
                scatters_info['size'] = [s_dict['size'] for _ in range(data.shape[0])]
            else:
                scatters_info['size'] = s_dict['size']
            # color
            if s_dict['color'] == '':
                scatters_info['color'] = ['black' for _ in range(data.shape[0])]
            elif s_dict['color'] == 'same':       
                scatters_info['color'] = color_list
            else:
                scatters_info['color'] = s_dict['color']
            # opacity
            if type(s_dict['alpha']) == float:
                scatters_info['alpha'] = [s_dict['alpha'] for _ in range(data.shape[0])]
            else: 
                scatters_info['alpha'] = s_dict['alpha']
            # shape
            if type(s_dict['shape']) == str:
                scatters_info['shape'] = [s_dict['shape'] for _ in range(data.shape[0])]
            else:
                scatters_info['shape'] = s_dict['shape']

            scatter_info_list.append(scatters_info)
            

    
    fig = go.Figure()
    for cond in range(data.shape[0]):
        fig.add_trace(go.Scatter3d(x=data[cond, :, 0], y=data[cond, :, 1], z=data[cond, :, 2], mode='lines', opacity=alpha, line=dict(color=color_list[cond])))

    # add scatter points if provided
    if scatters_dict is not None:
        for scatters_info in scatter_info_list:
            for cond in range(data.shape[0]):
                fig.add_trace(go.Scatter3d(
                    x=[data[cond, scatters_info['time'], 0]],
                    y=[data[cond, scatters_info['time'], 1]],
                    z=[data[cond, scatters_info['time'], 2]],
                    mode='markers',
                    marker=dict(
                        size=scatters_info['size'][cond],
                        color=scatters_info['color'][cond],
                        opacity=scatters_info['alpha'][cond],
                        symbol=scatters_info['shape'][cond]
                    )
                ))
   
    fig.update_layout(title=title,
                      xaxis_title=x_label,
                      yaxis_title=y_label)

    # change figure size and theme
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        template=template
    )

    # force axis to be equal
    if force_equal_axis:
        fig.update_layout(
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1
            )
        )
    
    # remove legends
    if not legend_show:
        fig.update_layout(showlegend=False)

    # save figure
    if save_path is not None:
        fig.write_image(save_path)
    
    if show:
        fig.show()