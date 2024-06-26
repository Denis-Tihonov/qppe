import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

############################################################################################################
def plot_short_timeseries(s, frequency, path = None):
    """Plot and save example of time series.

    Parameters
    ----------
    s : np.array
        Training time series.

    frequency : int
        Frequency, the number of repetitions per second.

    path : str or path-like
        See for fname in  matplotlib.pyplot.savefig.
    """ 
    plt.plot(np.arange(0,len(s))/frequency, s)
    plt.xlabel('$t,c$', size=20)
    plt.ylabel('$s(t)$', size=20)
    plt.tight_layout()
    
    if path is not None:
        plt.savefig(path, format='png', dpi=200)
        
    plt.show()

############################################################################################################
def plot_phase_trajectory(
    phase_trajectory,
    path = None,
    rotation = (0,0,0)
):
    """Plot and save example of phase trajectory.

    Parameters
    ----------
    phase_trajectory : np.array with shape (n_samples, 3)
        Training time series.

    path : str or path-like
        See for fname in  matplotlib.pyplot.savefig

    rotation : np.array with shape (3)
        Degree to rotate over axis.
    """ 
    ax = rotation[0]/180 * np.pi
    ay = rotation[1]/180 * np.pi
    az = rotation[2]/180 * np.pi
    
    T_X = np.array([[1,0,0],
                    [0,np.cos(ax),-np.sin(ax)],
                    [0,np.sin(ax), np.cos(ax)]])
    
    T_Y = np.array([[np.cos(ay),-np.sin(ay),0],
                    [np.sin(ay), np.cos(ay),0],
                    [0,0,1]])
    
    T_Z = np.array([[ np.cos(az),0,np.sin(az)],
                    [ 0,1,0],
                    [-np.sin(az),0,np.cos(az)]])
    
    phase_trajectory = phase_trajectory@T_Z@T_Y@T_X
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(phase_trajectory[:,0],phase_trajectory[:,1],phase_trajectory[:,2],lw = 1)
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=20, azim=135)
    
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, format='png', dpi=200, bbox_inches='tight')
        
    fig.show()

############################################################################################################
def plot_phase_trajectory_and_phase(
    phase_trajectory,
    expextation_values,
    phase_history,
    path = None,
    rotation = (0,0,0)
):
    """Plot and save example of phase trajectory with phase.

    Parameters
    ----------
    phase_trajectory : np.array with shape (n_samples, 3)
        Training time series.

    expextation_values : np.array with shape (n_samples, 3)
        Expextation model for phase trajectory of initial time series.

    phase_history : np.array with shape (n_samples)
        Phase of initial time series.

    path : str or path-like
        See for fname in  matplotlib.pyplot.savefig

    rotation : np.array with shape (3)
        Degree to rotate over axis.
    """ 
    ax = rotation[0]/180 * np.pi
    ay = rotation[1]/180 * np.pi
    az = rotation[2]/180 * np.pi
    
    T_X = np.array([[1,0,0],
                    [0,np.cos(ax),-np.sin(ax)],
                    [0,np.sin(ax), np.cos(ax)]])
    
    T_Y = np.array([[np.cos(ay),-np.sin(ay),0],
                    [np.sin(ay), np.cos(ay),0],
                    [0,0,1]])
    
    T_Z = np.array([[ np.cos(az),0,np.sin(az)],
                    [ 0,1,0],
                    [-np.sin(az),0,np.cos(az)]])
    
    phase_trajectory = phase_trajectory@T_Z@T_Y@T_X
    expextation_values = expextation_values@T_Z@T_Y@T_X
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  
    img =ax.scatter(
        phase_trajectory[:,0],
        phase_trajectory[:,1],
        phase_trajectory[:,2],
        c=cm.rainbow(phase_history[:]/(2*np.pi)),
        alpha = 1,
        s=5
    )
    
    m = cm.ScalarMappable(cmap=cm.rainbow)
    m.set_array(phase_history)
    
    cbar = fig.colorbar(m,shrink=0.5)
    
    cbar.ax.get_yaxis().set_ticks([])
    for i, lab in enumerate(['$0$','$\pi$/2','$\pi$','3$\pi$/2','2$\pi$']):
        cbar.ax.text(1, (3 * np.pi * i / 6), lab, size=16)
    
    ax.plot(
        expextation_values[:,0],
        expextation_values[:,1],
        expextation_values[:,2],
        color = 'orange',
        lw = 7
    )
    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=20, azim=135)
    
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    if path is not None:
        fig.savefig(path, format='png', dpi=200, bbox_inches='tight')
    
    fig.show()