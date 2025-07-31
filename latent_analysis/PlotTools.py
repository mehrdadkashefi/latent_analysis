import plotly.express as px
import matplotlib.pyplot as plt


def plot_RDM(x_dis, w=300, h=300, colormap = 'inferno', **kwargs):
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
