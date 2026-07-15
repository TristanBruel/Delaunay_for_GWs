import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
import argparse

from scipy import stats


class MixtureModel:
    """Simple mixture of two scipy distributions."""
    
    def __init__(self, dist1, dist2, weight1):
        """
        Initialize mixture model.
        
        Parameters:
        -----------
        dist1, dist2 : scipy.stats distribution objects
            Two scipy distributions to mix
        weight1 : float
            Weight for first distribution (0 < weight1 < 1)
            Weight for second distribution will be (1 - weight1)
        """
        self.dist1 = dist1
        self.dist2 = dist2
        self.weight1 = weight1
        self.weight2 = 1 - weight1
    
    def rvs(self, size):
        # Determine how many samples from each distribution
        n_from_dist1 = np.random.binomial(size, self.weight1)
        n_from_dist2 = size - n_from_dist1
        
        # Sample from each distribution
        samples1 = self.dist1.rvs(size=n_from_dist1) if n_from_dist1 > 0 else np.array([])
        samples2 = self.dist2.rvs(size=n_from_dist2) if n_from_dist2 > 0 else np.array([])
        
        # Combine and shuffle
        all_samples = np.concatenate([samples1, samples2])
        np.random.shuffle(all_samples)
        
        return all_samples
    
    def pdf(self, x):
        """
        Evaluate the probability density function of the mixture.
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the PDF
            
        Returns:
        --------
        numpy.ndarray
            PDF values at the given points
        """
        x = np.asarray(x)
        return self.weight1 * self.dist1.pdf(x) + self.weight2 * self.dist2.pdf(x)
    
    def cdf(self, x):
        """
        Evaluate the cumulative distribution function of the mixture.
        
        Parameters:
        -----------
        x : array_like
            Points at which to evaluate the CDF
            
        Returns:
        --------
        numpy.ndarray
            CDF values at the given points
        """
        x = np.asarray(x)
        return self.weight1 * self.dist1.cdf(x) + self.weight2 * self.dist2.cdf(x)


def generate_pop(mu1, cov1, mu2, cov2, weight1=0.5):
    """
    Generate a population that is a mixture model of two multivariate normal dist (scipy).

    Parameters:
    -----------
    mu1 : array
        Mean of the first distribution
    cov1 : array
        Symmetric positive (semi)definite covariance matrix of the first distribution
    mu2 : array
        Mean of the second distribution
    cov2 : array
        Symmetric positive (semi)definite covariance matrix of the second distribution

    Returns:
    --------
    MixtureModel
    """
    pop = MixtureModel(
        dist1 = stats.multivariate_normal(mu1, cov1),
        dist2 = stats.multivariate_normal(mu2, cov2),
        weight1 = weight1,
    )
    return pop


def plot_pdf(pop, plot3d=False, outfile=None):
    """
    Plot the 2d pdf of the population.
    """

    #####################################
    # Plotting parameters
    fs = 12
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs

    cmap = plt.get_cmap('viridis')

    if plot3d:
    # 3D plot
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
    else:
    # 2D plot
        fig, ax = plt.subplots(1, 1, figsize=(6,6))

    xgrid = np.linspace(-10,10,201)
    ygrid = np.linspace(-10,10,201)
    X, Y = np.meshgrid(xgrid, ygrid)
    pdf = pop.pdf(np.array([X.flatten(),Y.flatten()]).T)
    pdf = pdf.reshape(X.shape)
    
    if plot3d:
        surf = ax.plot_surface(X, Y, pdf, 
                               rstride=1, cstride=1, cmap=cmap, edgecolor='none',
                               )
        ax.set_zticks([])
        cbar = fig.colorbar(surf, shrink=0.5)
        cbar.set_label(r'pdf')
    else:
        ax.contourf(X, Y, pdf,
                    cmap=cmap,
                    levels=100,
                   )
        ax.contour(X, Y, pdf, colors='k')

    ax.set_xlabel(r'x')
    ax.set_xlim(-10,10)
    ax.set_ylabel(r'y')
    ax.set_ylim(-10,10)
    ax.set_title(r"`Astro' Population")

    plt.savefig(outfile, bbox_inches='tight', dpi=1200)

    return fig


def p_det(events, sigma=3):
    return np.exp(-np.sqrt(events[..., 0] ** 2 + events[..., 1] ** 2) / sigma)


def plot_pdet(pdet, plot3d=False, outfile=None):
    """
    Plot the 2d pdf of the detection probability.
    """

    #####################################
    # Plotting parameters
    fs = 12
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs

    cmap = plt.get_cmap('viridis')

    if plot3d:
    # 3D plot
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
    else:
    # 2D plot
        fig, ax = plt.subplots(1, 1, figsize=(6,6))

    xgrid = np.linspace(-10,10,201)
    ygrid = np.linspace(-10,10,201)
    X, Y = np.meshgrid(xgrid, ygrid)
    data = pdet(np.array([X.flatten(),Y.flatten()]).T)
    data = data.reshape(X.shape)

    if plot3d:
        surf = ax.plot_surface(X, Y, data,
                               rstride=1, cstride=1, cmap=cmap, edgecolor='none',
                               )
        ax.set_zticks([])
        cbar = fig.colorbar(surf, shrink=0.5)
        cbar.set_label(r'pdf')
    else:
        ax.contourf(X, Y, data,
                    cmap=cmap,
                    levels=100,
                   )
        ax.contour(X, Y, data, colors='k')


    ax.set_xlabel(r'x')
    ax.set_xlim(-10,10)
    ax.set_ylabel(r'y')
    ax.set_ylim(-10,10)
    ax.set_title(r'Detection probability')

    plt.savefig(outfile, bbox_inches='tight', dpi=1200)

    return fig



def sample_events(pop, Nevents, Nsamples, p_det, outfile_events, outfile_samples):
    """
    Generate samples for a series of event from a given population.

    Parameters:
    -----------
    pop : 
        Model for the 'astro' population
    Nevents : int
        Number of events simulated
    Nsamples : int
        Number of samples per event

    Returns:
    --------
    2D array
       Samples representing measurements for the observed events
    """
    if os.path.exists(outfile_events) and os.path.exists(outfile_samples):
        print('Loading observed events from:', outfile_events)
        observed_events = np.loadtxt(outfile_events)
        print('Loading samples from:', outfile_samples)
        samples = np.loadtxt(outfile_samples)
    else:
        print('Generating samples.')
        events = pop.rvs(size=Nevents)
        observed_events = events[np.random.random(events.shape[0]) < p_det(events)]
        np.savetxt(outfile_events, observed_events)
        # For now we set the distribution of errors as a 2D gaussian
        distro_errors = stats.multivariate_normal(np.zeros(2), 1e-2*np.eye(2))
        #samples = np.repeat(observed_events, Nsamples, axis=0) + distro_errors.rvs(size=observed_events.shape[0]*Nsamples)
        shifted = observed_events + distro_errors.rvs(size=observed_events.shape[0])
        samples = np.repeat(shifted, Nsamples, axis=0) + distro_errors.rvs(size=shifted.shape[0]*Nsamples)
        np.savetxt(outfile_samples, samples)

    print('Number of detected events:', len(observed_events))
    return observed_events, samples


def plot_samples(observed_events, samples, Nsamples, outfile):
    """
    Plot events.
    """
    event_limits = np.arange(0,len(samples),Nsamples)
    event_barycenters = (
            np.add.reduceat(samples, event_limits, axis=0) /Nsamples
            )


    #####################################
    # Plotting parameters
    fs = 12
    lw = 1.4
    plt.rcParams['font.size']=fs
    plt.rcParams['font.family']='serif'
    plt.rcParams['font.serif']='cmr10'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.unicode_minus']=False
    plt.rcParams['axes.formatter.use_mathtext']=True
    plt.rcParams['lines.linewidth']=lw
    plt.rcParams['xtick.labelsize']=fs
    plt.rcParams['ytick.labelsize']=fs
    plt.rcParams['legend.fontsize']=.9*fs

    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(1, 1, figsize=(6,6))

    ax.scatter(observed_events[:,0], observed_events[:,1],
            marker='.', s=2**2, c='r', zorder=100,
            label='Observed events (real)',
            )

    ax.scatter(event_barycenters[:,0], event_barycenters[:,1], 
            marker='*', s=7**2, c='k',
            label='Samples (barycenters)',
            )

    ax.set_xlabel(r'x')
    ax.set_xlim(-10,10)
    ax.set_ylabel(r'y')
    ax.set_ylim(-10,10)
    ax.set_title(r'Events and samples')
    ax.legend(loc='upper right')

    plt.savefig(outfile, bbox_inches='tight', dpi=1200)

    return fig


##################################################################
### Run it!
### 
##################################################################
if __name__ == "__main__":

    work_dir = './'
    plot_dir = os.path.join(work_dir,'plots')

    # Define command line options
    parser = argparse.ArgumentParser()
    # Set 'astro' population
    parser.add_argument("--mu1", dest='mu1', help="Mean of first distribution", default=np.array([-5,0]))
    parser.add_argument("--cov1", dest='cov1', help="Covariance matrix of first distribution", 
                        default=np.array([[3,1],[1,3]]),
                        )
    parser.add_argument("--mu2", dest='mu2', help="Mean of second distribution", default=np.array([5,0]))
    parser.add_argument("--cov2", dest='cov2', help="Covariance matrix of second distribution", 
                        default=np.array([[2,-1],[-1,1]]),
                        )
    # Set detection probability
    parser.add_argument("--sigma", dest='sigma_det', help="Exponential parameter of the detection probability", type=float, default=3)
    # Events and samples
    parser.add_argument("--events", dest='Nevents', help="Number of events", type=int, default=1_000)
    parser.add_argument("--samples", dest='Nsamples', help="Number of samples per event", type=int, default=10_000)
    # Show the plots
    parser.add_argument("-p", dest='show_plots', action='store_true', help="Show plots")
    # Plot pdf as 3D distribution
    parser.add_argument("--3D", dest='plot3d', action='store_true', help="Plot pdf in 3D")
    args = parser.parse_args()

    # Generate 'astro' pop
    pop = generate_pop(args.mu1,args.cov1,args.mu2,args.cov2)

    # Sample observable events
    pdet = lambda events: p_det(events, sigma=args.sigma_det)
    filename = 'observed_events.txt'
    outfile_events = os.path.join(work_dir,filename)
    filename = 'samples.txt'
    outfile_samples = os.path.join(work_dir,filename)
    observed_events, samples = sample_events(pop, args.Nevents, args.Nsamples, pdet, 
            outfile_events, outfile_samples,
            )

    if args.show_plots:
        # Plot astro pop
        filename = 'pdf_pop' + args.plot3d*'_3D' + '.png'
        outfile = os.path.join(plot_dir,filename)
        fig_pop = plot_pdf(pop, args.plot3d, outfile)
        # Plot detection probability
        filename = 'pdet' + args.plot3d*'_3D' + '.png'
        outfile = os.path.join(plot_dir,filename)
        fig_pdet = plot_pdet(pdet, args.plot3d, outfile)
        # Plot samples from observed events
        filename = 'events%i_samples%i.png' %(args.Nevents,args.Nsamples)
        outfile = os.path.join(plot_dir,filename)
        fig_samples = plot_samples(observed_events, samples, args.Nsamples, outfile)

        plt.show()
