import numpy as np
import scipy.stats as stats
import scipy.special as special
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def draw_thetas(N):
   
    cos_thetas = np.random.uniform(-1, 1, N)
    cos_incs = np.random.uniform(-1, 1, N)
    phis = np.random.uniform(0, 2*np.pi, N)
    zetas = np.random.uniform(0, 2*np.pi, N)

    Fps1 = 0.5*np.cos(2*zetas)*(1 + cos_thetas**2)*np.cos(2*phis) - np.sin(2*zetas)*cos_thetas*np.sin(2*phis)
    Fxs1 = 0.5*np.sin(2*zetas)*(1 + cos_thetas**2)*np.cos(2*phis) + np.cos(2*zetas)*cos_thetas*np.sin(2*phis)

    return np.sqrt(0.25*(Fps1**2)*(1 + cos_incs**2)**2 + (Fxs1**2)*cos_incs**2)

def gaussian(x,  mean, std):

    return (1./(np.sqrt(2.*np.pi)*std))*np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    # if mean.ndim==1:

    #     return (1./(np.sqrt(2.*np.pi)*std[:,None]))*np.exp(-((x[None, :] - mean[:, None]) ** 2) / (2 * std[:, None] ** 2))

    # elif mean.ndim==2:

    #     return (1./(np.sqrt(2.*np.pi)*std))*np.exp(-((x[None, :] - mean) ** 2) / (2 * std** 2))

def gaussian_truncated(x, mean, std, mlow=2, mhigh=100):

    gauss=gaussian(x, mean, std)

    # breakpoint()

    fact=0.5*(special.erf((mhigh-mean)/(np.sqrt(2)*std))+special.erf((mean-mlow)/(np.sqrt(2)*std)))

    ans=gauss/fact

    ans[x<mlow]=0

    ans[x>mhigh]=0

    ans[fact==0]=0.

    # if mlow.ndim==1 and mhigh.ndim==1 and mean.ndim==1:

    #     fact=0.5*(special.erf((mhigh-mean)/(np.sqrt(2)*std))+special.erf((mean-mlow)/(np.sqrt(2)*std)))

    #     ans=gauss/fact[:,None]

    #     ans[x[None,:]<mlow[:,None]]=0

    #     ans[x[None,:]>mhigh[:,None]]=0

    #     ans[fact==0]=0.

    # else:

    #     if mlow.ndim==1:
    #         mlow=mlow[:,None]
    #     if mhigh.ndim==1:
    #         mhigh=mhigh[:,None]
    #     if mean.ndim==1:
    #         mean=mean[:,None]
    #         ampl=ampl[:,None]
    #         std=std[:,None]

    #     fact=0.5*(special.erf((mhigh-mean)/(np.sqrt(2)*std))+special.erf((mean-mlow)/(np.sqrt(2)*std)))

    #     ans=gauss/fact

    #     ans[x[None,:]<mlow]=0

    #     ans[x[None,:]>mhigh]=0

    #     ans[fact==0]=0.


    ans[gauss==0]=0



    return ans

def p_broken_powerlaw(m1, alpha_1, alpha_2, m_break, m1_low, m_high):
    m1 = np.array(m1)

    pdf = np.zeros_like(m1, dtype=float)
    mask1 = (m1_low <= m1) & (m1 < m_break)
    mask2 = (m_break <= m1) & (m1 <= m_high)
    
    pdf[mask1] = (m1[mask1] / m_break) ** (-alpha_1)
    pdf[mask2] = (m1[mask2] / m_break) ** (-alpha_2)

    #normalization constant
    term1 = (1 - (m1_low/m_break)**(1-alpha_1)) / (1 - alpha_1)
    term2 = ((m_high/m_break)**(1-alpha_2) - 1) / (1 - alpha_2)
    N = m_break * (term1 + term2)

    return pdf / N

def smoothing_function(m, m_low, delta_m):
    """
    S(m | m_low, delta_m) from Eq. (B21).
    
    Parameters
    ----------
    m : array-like
        Mass values.
    mlow : float
        Lower mass cutoff.
    delta_m : float
        Width of the smoothing region.
    """
    m = np.asarray(m)
    S = np.zeros_like(m, dtype=float)
    
    # region 1: m < mlow → 0 (already set)
    
    # region 2: transition region
    mask2 = (m_low <= m) & (m < m_low + delta_m)
    mprime = m[mask2] - m_low
    f = np.exp(delta_m / mprime + delta_m / (mprime - delta_m))
    S[mask2] = 1.0 / (1.0 + f)
    
    # region 3: m >= mlow + delta_m → 1
    mask3 = (m >= m_low + delta_m)
    S[mask3] = 1.0
    return S

def p_m1(m1, alpha_1, alpha_2, m_break, m1_low, m1_high, mu_1, sigma_1, mu_2, sigma_2, lamda_0, lamda_1, delta_m1, norm=False):
    norm_1 = gaussian_truncated(m1, mu_1, sigma_1, mhigh = m1_high, mlow=m1_low)
    norm_2 = gaussian_truncated(m1, mu_2, sigma_2, mhigh = m1_high, mlow=m1_low)
    lamda_2 = 1 - lamda_0 - lamda_1
    pbp = p_broken_powerlaw(m1, alpha_1, alpha_2, m_break, m1_low, m1_high)
    S = smoothing_function(m1, m1_low, delta_m1)

    dist = (lamda_0 * pbp + lamda_1 * norm_1 + lamda_2 * norm_2) * S

    if norm:

        m1_grid=np.linspce(m1_low,m1_high,1000)

        norm_1_grid = gaussian_truncated(m1, mu_1, sigma_1, mhigh = m1_high, mlow=m1_low)
        norm_2_grid = gaussian_truncated(m1, mu_2, sigma_2, mhigh = m1_high, mlow=m1_low)
        
        pbp_grid = p_broken_powerlaw(m1, alpha_1, alpha_2, m_break, m1_low, m1_high)
        S_grid = smoothing_function(m1, m1_low, delta_m1)

        dist_grid = (lamda_0 * pbp_grid + lamda_1 * norm_1_grid + lamda_2 * norm_2_grid) * S_grid

        norm = np.trapezoid(dist_grid, m1_grid)

        dist/=norm

    return dist 


def draw_m1(nsamp,alpha_1, alpha_2, m_break, m1_low, m1_high, mu_1, sigma_1, mu_2, sigma_2, lamda_0, lamda_1, delta_m1):

    m1_grid=np.linspace(m1_low,m1_high,1000)

    pdfs_m1=p_m1(m1_grid, alpha_1, alpha_2, m_break, m1_low, m1_high, mu_1, sigma_1, mu_2, sigma_2, lamda_0, lamda_1, delta_m1, norm=False)

    pdf_max=np.amax(pdfs_m1)

    samples=np.zeros(nsamp)

    for i in range(nsamp):

        to_draw=True

        while to_draw:
            m1_try=np.random.uniform(m1_low,m1_high)

            pdf_try=p_m1(np.array([m1_try]), alpha_1, alpha_2, m_break, m1_low, m1_high, mu_1, sigma_1, mu_2, sigma_2, lamda_0, lamda_1, delta_m1, norm=False)

            if pdf_try>pdf_max*np.random.uniform(0,1):

                to_draw=False

        samples[i]=m1_try


    return samples


def draw_m1_cdf(nsamp,alpha_1, alpha_2, m_break, m1_low, m1_high, mu_1, sigma_1, mu_2, sigma_2, lamda_0, lamda_1, delta_m1):

    m1_grid=np.linspace(m1_low,m1_high,10000)

    pdfs_m1=p_m1(m1_grid, alpha_1, alpha_2, m_break, m1_low, m1_high, mu_1, sigma_1, mu_2, sigma_2, lamda_0, lamda_1, delta_m1, norm=False)

    cdfs=np.cumsum(pdfs_m1)
    cdfs/=cdfs[-1]

    inv_cdf_spline=spline(cdfs,m1_grid)

    samples=inv_cdf_spline(np.random.uniform(0,1,nsamp))

    # breakpoint()

    return samples


def draw_pl(nsamp,alpha, m1_low, m1_high):

    ys=np.random.uniform(0,1,nsamp)

    samples=((m1_high**(alpha+1)-m1_low**(alpha+1))*ys+m1_low**(alpha+1))**(1./(alpha+1))

    return samples

def power_law_pdf(xs,alpha, m1_low, m1_high):

    return (alpha+1)*xs**(alpha)/(m1_high**(alpha+1)-m1_low**(alpha+1))



def z_pdf_not_norm(zs,kappas,dVc_spl,zmax):

    ans=(1+zs)**(kappas-1)*np.asarray(dVc_spl(zs))

    ans[zs>zmax]=0


    
    # try:
    #     ans=(1+zs[None,:])**(kappas[:,None]-1)*np.asarray(dVc_spl(zs.get()))[None,:]
    # except AttributeError:
    #     ans=(1+zs[None,:])**(kappas[:,None]-1)*np.asarray(dVc_spl(zs))[None,:]

    # z_ext=np.repeat(zs,len(kappas)).reshape((len(zs),len(kappas))).T


    # ans[z_ext>zmax]=0

    return ans


def z_pdf(zs,kappas,dVc_spl,zmax):

    z_grid=np.linspace(0,zmax,1000)
    pdf_grid=z_pdf_not_norm(z_grid,kappas,dVc_spl,zmax)
    norm=np.trapezoid(pdf_grid,x=z_grid)

    ans=z_pdf_not_norm(zs,kappas,dVc_spl,zmax)/norm

    return ans

def draw_z(nsamp,kappa,dVc_spl,zmax):

    z_grid=np.linspace(0,zmax,1000)

    pdfs_z=z_pdf_not_norm(z_grid, kappa,dVc_spl,zmax)

    pdf_max=np.amax(pdfs_z)

    samples=np.zeros(nsamp)

    for i in range(nsamp):

        to_draw=True

        while to_draw:
            z_try=np.random.uniform(0,zmax)

            pdf_try=z_pdf_not_norm(np.array([z_try]), kappa,dVc_spl,zmax)

            if pdf_try>pdf_max*np.random.uniform(0,1):

                to_draw=False

        samples[i]=z_try


    return samples


def draw_z_cdf(nsamp,kappa,dVc_spl,zmax):

    z_grid=np.linspace(0,zmax,10000)

    pdfs_z=z_pdf_not_norm(z_grid, kappa,dVc_spl,zmax)

    cdfs=np.cumsum(pdfs_z)
    cdfs/=cdfs[-1]

    inv_cdf_spline=spline(cdfs,z_grid)

    samples=inv_cdf_spline(np.random.uniform(0,1,nsamp))

    return samples




def pdf_q(qs,m1,mg_low,mg_up,m_min,betaq,norm=False):

    ans= qs**betaq

    ans[qs*m1<m_min]=0.
    ans[(qs*m1>mg_low) & (qs*m1<mg_up)]=0.

    if norm:

        norm=((mg_low/m1)**(betaq+1)-(m_min/m1)**(betaq+1)+1-(mg_up/m1)**(betaq+1))/(betaq+1.)
        

    return ans


def pdf_q_no_gap(qs, m1, m_min, betaq, norm=False):
    
    # Broadcast qs and m1 to same shape
    qs, m1 = np.broadcast_arrays(qs, m1)

    ans = qs**betaq
    ans = np.where(qs * m1 < m_min, 0., ans)

    if norm:
        # Normalization factor depends on m1 elementwise
        norm_factor = (1. - (m_min / m1)**(betaq + 1)) / (betaq + 1.)
        ans = ans / norm_factor

    return ans


def draw_q(nsamp,m1s,mg_low,mg_up,m_min,betaq):

    q_grid=np.linspace(0,1,1000)

    samples=np.zeros(nsamp)

    for i in range(nsamp):

        if betaq>0:
        

            pdf_max=pdf_q(np.array([1.]),m1s[i],mg_low,mg_up,m_min,betaq,norm=False)

        else:

            pdf_max=pdf_q(np.array([m_min/m1s[i]]),m1s[i],mg_low,mg_up,m_min,betaq,norm=False)
            


        to_draw=True

        while to_draw:

            q_try=np.random.uniform(0,1)

            pdf_try = pdf_q(np.array([q_try]),m1s[i],mg_low,mg_up,m_min,betaq,norm=False)

            if pdf_try>pdf_max*np.random.uniform(0,1):
                to_draw=False

        samples[i]=q_try

    return samples


def draw_q_no_gap(nsamp,m1s,m_min,betaq):

    ys=np.random.uniform(0,1,nsamp)

    samples=((1-(m_min/m1s)**(betaq+1))*ys+(m_min/m1s)**(betaq+1))**(1./(betaq+1))

    return samples



def draw_events(num_events,params,settings,q_no_gap=False):

    events0=np.zeros((num_events,3))

    
    events0[:,0]=draw_m1(num_events,params['alpha_1'], params['alpha_2'], params['m_break'], params['m1_low'], params['m1_high'], params['mu_1'], params['sigma_1'], params['mu_2'], params['sigma_2'], params['lamda_0'], params['lamda_1'], params['delta_m1'])
    # print('m1 done')

    # breakpoint()

    events0[:,2]=draw_z(num_events,params['kappa'],settings['dVc_spl'],settings['zmax'])
    # print('z done')

    if q_no_gap:
        events0[:,1]=draw_q_no_gap(num_events,events0[:,0],params['m1_low'],params['betaq'])
    else:
        events0[:,1]=draw_q(num_events,events0[:,0],params['mg_low'],params['mg_up'],params['m1_low'],params['betaq'])
    
    # print('m2 done')
    

    return events0

def broken_power_law_pdf(x,beta,index1,index2,xbreak,xlow,xup,norm=False):

    ans=(x**(index1)+beta)/(1.+(x/xbreak)**(index1-index2))

    # breakpoint()
    ans[x<xlow]=0
    ans[x>xup]=0

    if norm:

        xgrid=np.linspace(xlow,xup,1000)

        ans_grid=(xgrid**(index1)+beta)/(1.+(xgrid/xbreak)**(index1-index2))

        norm=np.trapezoid(ans_grid,x=xgrid)

        ans=ans/norm


    return ans

def draw_broken_power_law(npts,params,settings):

    zgrid=np.linspace(0,settings['zmax'],1000)

    pdf_grid=broken_power_law_pdf(zgrid,params['beta'],params['index1'],params['index2'],params['zbreak'],settings['zmin'],settings['zmax'],norm=False)

    pdf_max=np.amax(pdf_grid)

    samples=np.zeros(npts)

    for i in range(npts):

        to_draw=True

        while to_draw:

            sample_test=np.random.uniform(0,settings['zmax'])
            
            pdf_test=broken_power_law_pdf(np.array([sample_test]),params['beta'],params['index1'],params['index2'],params['zbreak'],settings['zmin'],settings['zmax'],norm=False)[0]

            if pdf_test>pdf_max*np.random.uniform(0,1):

                samples[i]=sample_test

                to_draw=False

    return samples


def hist_pdf(xs,xhist,xend,xmin,xmax,impose_min_width=1,min_width=0.1,log_hist=0):

    edges=xhist[:,0]

    values=xhist[:,1]

    # values_ends=xends

    nmiddle=len(edges)

    

    values=values[np.argsort(edges)]
    edges=edges[np.argsort(edges)]

    edges_all=np.zeros(nmiddle+2)
    values_all=np.zeros(nmiddle+1)

    edges_all[1:-1]=np.copy(edges)
    edges_all[0]=xmin
    edges_all[-1]=xmax

    # print(edges_all)

    values_all[:-1]=np.copy(values)
    values_all[-1]=xend[0]
    # counts,bin0=np.histogram(data,edges_all)

    if log_hist==1:
        values_all=10**values_all

    # breakpoint()
    inds_bin=np.digitize(xs,edges_all)-1

    inds_bin=np.clip(inds_bin,0,nmiddle)

    diffs=edges_all[1:]-edges_all[:-1]

    if impose_min_width==1:
        if np.sum(diffs<min_width)>0:
            check=0
            return 0,0,0
        
    pdfs=values_all[inds_bin]

    norm=np.sum(values_all*diffs)



    return pdfs,norm,1

def draw_hist(npts,xhist,xend,settings):

    zgrid=np.linspace(0,settings['zmax'],1000)

    pdf_grid=hist_pdf(zgrid,xhist,xend,settings['zmin'],settings['zmax'],impose_min_width=0)[0]

    pdf_max=np.amax(pdf_grid)

    samples=np.zeros(npts)

    for i in range(npts):

        to_draw=True

        while to_draw:

            sample_test=np.random.uniform(0,settings['zmax'])
            
            pdf_test=hist_pdf(np.array([sample_test]),xhist,xend,settings['zmin'],settings['zmax'],impose_min_width=0)[0][0]

            if pdf_test>pdf_max*np.random.uniform(0,1):

                samples[i]=sample_test

                to_draw=False

    return samples


def gauss_pdf(x,mu,sigma):

    return (1./(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*(x-mu)**2/sigma**2)
