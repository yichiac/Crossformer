# Stiff Circuit System Modeling via Transformer

Weiman Yan, Yi-Chia Chang, Wanyu Zhao, Liam Li

## Dataset

The dataset `datasets/adc/ft/adc.csv` contains 2000 data records. Each data record consists of 5 rows where the first three rows are input signals and the last two rows are output signals. The dataset is obtained by running SPICE transient simulation on an ADC circuit. Input signal is generated using PRBS sequences and output signal is sampled every 2.5 ns.

## Experiment result
<p float="left">
<img src=".\figs\loss.png" height = "300" alt="" align=center />
<img src=".\figs\prediction.png" height = "300" alt="" align=center />

<b>Figure 1.</b> The loss curves vs epochs (left) and one example fitting result using trained model.
</p>

## Reproduce experiment result

To reproduce the result, run

```
python run_nrmse.py
```

To use our model on the dataset, change `config.json` and our top-level `run_nrmse.py`.

## Hardware requirements
GPU: NVIDIA A100 GPU 40 GB

Memory: 16 GB

Number of CPUs: 8
