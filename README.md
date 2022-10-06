# Fair Wrappers for Black-box Predictions
Python implementation of "Fair Wrappers for Black-box Predictions"

# Code SI for "Fair Wrappers for Black-box Predictors"
General code for the TopDown algorithm.

## Installation
We configure using conda. `environment.yml` provides dependencies to run the code.
For instance:

```
conda env create -f environment.yml
```

Additional files need to be setup and downloaded for AIF360 and ACS/Folktables datasets to work.

## Demo
`main.py` is used to generate main-text results. An example:

```
ipython main.py -- --dataset acs --blackbox rf --calibrate --min-split 0.1 --min-leaf 30 --num-splits 32 --save-folder experiments/acs/conservative
ipython main.py -- --dataset acs --blackbox rf --calibrate --equality-of-opportunity-mode -K 2 --epsilon 0.05 --min-split 0.1 --min-leaf 30 --num-splits 32 --save-folder experiments/acs/conservative_eoo
ipython main.py -- --dataset acs --blackbox rf --calibrate --statistical-parity-mode --min-split 0.1 --min-leaf 30 --num-splits 32 --save-folder experiments/acs/conservative_sp

ipython main.py -- --dataset acs --blackbox rf --calibrate --min-split 0.1 --min-leaf 30 --num-splits 32 --save-folder experiments/acs/aggressive --aggressive-update
ipython main.py -- --dataset acs --blackbox rf --calibrate --equality-of-opportunity-mode -K 2 --epsilon 0.05 --min-split 0.1 --min-leaf 30 --num-splits 32 --save-folder experiments/acs/aggressive_eoo --aggressive-update
ipython main.py -- --dataset acs --blackbox rf --calibrate --statistical-parity-mode --min-split 0.1 --min-leaf 30 --num-splits 32 --save-folder experiments/acs/aggressive_sp --aggressive-update

```
See `--help` options for full set of flags.

Furthermore, `proxy.py` and `year_eval.py` are given to generate plots as per the Appendix / SI sections regarding proxy sensitive attributes and distribution shift, respectively. The method of running these scripts are mostly identical to `main.py`, use `--help` for specific flags.

## Notes

Type annotations are just for readability.

The warning about the feature names from `sklearn`'s `ColumnTransformer` does not seem to be important. As far as I can see, it does not effect anything and might be a bug from `sklearn`.

`invalid value encountered in double_scalars` warnings are being suppressed when running `scalar_minimizer` from `scipy`. This also does not seem to be effecting anything. Warnings can be turned back on by changing the constant at the top of `nodefunction.py`.

## Reference

*Fair Wrapping for Black-box Predictions* <br>
Soen A, Alabdulmohsin I, Koyejo S, Mansour Y, Moorosi N, Nock R, Sun K, Xie L <br>
Conference on Neural Information Processing Systems, NeurIPS 2022