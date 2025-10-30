# zfitexample.py
# LEGACY -- just to remind me how to use zfit
import os
import sys
from glob import glob
from argparse import ArgumentParser

import uproot
import awkward as ak
import numpy as np

import tensorflow as tf
import zfit
#from zfit.minimize import DefaultToyStrategy
#zfit.run.numeric_checks = True
#zfit.settings.changed_warnings.errors_name = False
#zfit.settings.changed_warnings.hesse_name = False

import mplhep
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import config as cfg

parser = ArgumentParser()
parser.add_argument("--cut",       type=str, required=False, default = "", help='Cut expression to pass to uproot')
parser.add_argument("--model",     type=str, required=True,  help='Name of the pdf to fit to data')
parser.add_argument("--sample",    type=int, required=True,  help='Index of the sample to be used')
parser.add_argument("--inputpath", type=str, default='/r01/lhcb/rrm42/fcc/stage1_postBDT', help='Path to files containing cfg.samples')
parser.add_argument("--nchunks",   type=int, default=None, help='Number of chunks to run over')
args = parser.parse_args()

print('Initialising...')
print(f'Using {args.inputpath} with cut {args.cut}')
print(f"{30*'-'}")

sample = cfg.samples[args.sample]
path = os.path.join(args.inputpath, sample, '*.root')
files = glob(path)

if args.nchunks is not None:
    files = files[:args.nchunks]

data_np = np.array([])
#data = {sample: np.array([]) for sample in cfg.samples}

#for sample in cfg.samples:
#    path = os.path.join(args.inputpath, sample, '*.root')
#    files = glob(path)[:chunks[sample]]
#    for file in files:
#        with uproot.open(file) as f:
#            data[sample] = np.append(data, f['events'].arrays('45.5 - EVT_hemisEmin_e', cut = args.cut, library='np'))

for file in files:
    with uproot.open(file) as f:
        temp = f['events'].arrays(f'{0.5*cfg.mass_Z} - EVT_hemisEmin_e', cut=args.cut, library='np')[f'{0.5*cfg.mass_Z} - EVT_hemisEmin_e']
        data_np = np.append(data_np, temp)

obs    = zfit.Space('Emiss', cfg.EVT_hemisEmin_e_withpresel_min, cfg.EVT_hemisEmin_e_withpresel_max)
data = zfit.Data(data_np, obs=obs)

if args.model == 'DoubleCB':
    mu     = zfit.Parameter("mu",      40,   0, 80)
    sigma  = zfit.Parameter("sigma",   10,   0, 50)
    alphal = zfit.Parameter("alphal", -35, -50, 50)
    nl     = zfit.Parameter("nl",       3,   0, 10)
    alphar = zfit.Parameter("alphar",  40,   0, 80)
    nr     = zfit.Parameter("nr",       3,   0, 10)
    model = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)

elif args.model == 'CrystalBall':
    mu     = zfit.Parameter("mu",      40,   0, 80)
    sigma  = zfit.Parameter("sigma",   10,   0, 50)
    alpha = zfit.Parameter("alpha", -35, -50, 50)
    n     = zfit.Parameter("n",       3,   0, 10)
    model = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)

elif args.model == 'JohnsonSU':
    mu = zfit.Parameter("mu", 40)
    lambd = zfit.Parameter("lambd", 20)
    gamma = zfit.Parameter("gamma", 1)
    delta = zfit.Parameter("delta", 1)
    model = zfit.pdf.JohnsonSU(obs=obs, mu=mu, lambd=lambd, gamma=gamma, delta=delta)

elif args.model == 'KDE':
    model = zfit.pdf.KDE1DimISJ(data)

if args.model != 'KDE':
    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    #minimizer = zfit.minimize.Minuit(
    #    strategy        = DefaultToyStrategy,
    #    tol             = 1e-5,
    #    mode            = 2,
    #    use_minuit_grad = True,
    #    verbosity       = 0,
    #)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    param_errors, _ = result.errors()

    print(f"Function minimum: {result.fmin}")
    print(f"Converged: {result.converged}")
    print(f"Valid: {result.valid}")
    print(result.params)

bins = 100
mplhep.histplot(data.to_binned(bins), yerr=True, density=True, color='black', histtype='errorbar')
x_plot = np.linspace(obs.v1.lower, obs.v1.upper, num=1000)
y_plot = model.pdf(x_plot)
plt.plot(x_plot, y_plot, color='xkcd:blue')
plt.title(cfg.titles[cfg.samples[args.sample]])
plt.tight_layout()
plt.show()
