import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def fit_fp_decay(beta, energy, d_energy, save_plot=False):
    # fit the exponential energy decaying of free projection energy

    def exp_plateau(beta, E_inf, A, gamma):
        return E_inf + A * np.exp(-gamma * beta)

    # Initial guesses: E_inf ~ last points, A ~ E(0)-E_inf, gamma ~ 1
    p0 = [energy[-1], energy[0]-energy[-1], 1]

    popt, pcov = curve_fit(exp_plateau, beta, energy, p0=p0,
                        sigma=d_energy, absolute_sigma=True,
                        maxfev=10000)

    E_inf, A, gamma = popt
    perr = np.sqrt(np.diag(pcov))
    dE_inf, dA, dgamma = perr

    # Report
    print("=" * 80)
    print("  Exponential-Energy Cooling fit:  E(beta) = E_inf + A exp(-Gamma*beta)")
    print("=" * 80)
    print(f"  E_inf   = {E_inf:.6f} ± {dE_inf:.6f}")
    print(f"  A       = {A:.6f} ± {dA:.6f}")
    print(f"  Gamma   = {gamma:.4f} ± {dgamma:.4f}")
    print(f"  System cooled to about 37% initial Energy gap at beta = {1/gamma:.4f} a.u. (1/Gamma)")
    print(f"  System considered fully cooled at about beta = [{3/gamma:.4f}, {5/gamma:.4f}] a.u.")
    print("=" * 80)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                gridspec_kw={'height_ratios': [3, 1]},
                                sharex=True)

    beta_fine = np.linspace(0, beta[-1] * 1.05, 300)
    ax1.errorbar(beta, energy, yerr=d_energy, fmt='o', ms=4, capsize=3,
                color='C0', label='Data')
    ax1.plot(beta_fine, exp_plateau(beta_fine, *popt), '-', color='C1',
            label=rf'Fit: E($\beta$)={E_inf:.2f}+{A:.2f}exp(-{gamma:.2f}$\beta$)')
    ax1.axhline(E_inf, ls='--', color='C2', alpha=0.7,
                label=rf' $E_\infty$={E_inf:.5f}±{dE_inf:.5f}')
    ax1.fill_between(beta_fine, E_inf - dE_inf, E_inf + dE_inf,
                    color='C2', alpha=0.15)
    ax1.set_ylabel('Energy')
    ax1.set_title('Imaginary-time cooling curve')
    ax1.legend(fontsize=9)

    # Residuals
    residuals = (energy - exp_plateau(beta, *popt)) / d_energy
    ax2.errorbar(beta, residuals, yerr=1, fmt='o', ms=4, capsize=3, color='C0')
    ax2.axhline(0, ls='-', color='gray', lw=0.8)
    ax2.axhline(2, ls=':', color='gray', lw=0.6)
    ax2.axhline(-2, ls=':', color='gray', lw=0.6)
    ax2.set_xlabel(r'Imaginary time  $\beta$')
    ax2.set_ylabel(r'Residual ($\sigma$)')
    plt.tight_layout()
    if save_plot:
        plt.savefig('./fp_energy_decay.png', dpi=150)
    plt.close(fig) # don't show plot

    return E_inf, dE_inf