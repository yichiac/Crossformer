# Stiff Circuit System Modeling via Transformer

Weiman Yan, Yi-Chia Chang, Wanyu Zhao, Liam Li

## Dataset

The dataset `datasets/adc/ft/adc.csv` contains 2000 data records. Each data record consists of 5 rows where the first three rows are input signals and the last two rows are output signals. The dataset is obtained by running SPICE transient simulation on an ADC circuit. Input signal is generated using PRBS sequences and output signal is sampled every 2.5 ns.

## Experiment result
<p float="left">
<img src=".\figs\loss_kan_n10_grid5_k3.png" height = "300" alt="" align=center />
<img src=".\figs\prediction_kan_n10_grid5_k3.png" height = "300" alt="" align=center />

<b>Figure 1.</b> The loss curves vs epochs (left) and one example fitting result using trained model.
</p>

## Reproduce experiment result

To reproduce the result, run

```
python run.py --n 10 --grid 5 --k 3 --seed 0
```

To use our model on the dataset, change `config.json` and our top-level `run.py`.

## Hardware requirements
GPU: NVIDIA A100 GPU 80 GB

Memory: 16 GB

Number of CPUs: 8
