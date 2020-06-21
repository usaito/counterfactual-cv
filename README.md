## Counterfactual Cross-Validation

---

### About

This repository accompanies the semi-synthetic simulations conducted in the paper **"Counterfactual Cross-Validation: Stable Model Selection Procedure for Causal Inference Models"** by [Yuta Saito](https://usaito.github.io/) and [Shota Yasui](https://yasui-salmon.github.io/), which has been accepted by [ICML2020](https://icml.cc/).

If you find this code useful in your research then please cite:

```
@article{saito2019counterfactual,
  title={Counterfactual Cross-Validation: Effective Causal Model Selection from Observational Data},
  author={Saito, Yuta and Yasui, Shota},
  journal={arXiv preprint arXiv:1909.05299},
  year={2019}
}
```

### Dependencies

- numpy==1.16.2
- pandas==0.24.2
- scikit-learn==0.20.3
- tensorflow==1.15.2
- plotly==3.10.0
- econml==0.4
- optuna==0.19.0

### Running the code

To run the simulation with IHDP data, navigate to the `src/` directory and run the command

```
python main.py\
  --iters 100\
  --n_trials 100\
  --alpha_list 100 50 10 5 1 0.5 0.1 0.05 0.01
```

This will run semi-synthetic experiments (model selection and hyper-parameter tuning) conducted in Section 5 of the paper.

## Reference
1. Shalit, U., Johansson, F. D., and Sontag, D. "Estimating individual treatment effect: generalization bounds and algorithms," In *Proceedings of the 34th International Conference on Machine Learning*-Volume 70, pp. 3076–3085. JMLR. org, 2017. [github](https://github.com/clinicalml/cfrnet)
