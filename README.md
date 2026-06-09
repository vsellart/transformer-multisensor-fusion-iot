# transformer-multisensor-fusion-iot

Bachelor's thesis on transformer-based multisensor fusion for energy-efficient sensor-to-edge transmission in IoT networks.

The project trains a Transformer model to reconstruct original byte sequences from corrupted multisensor observations and evaluates a confidence-based masking strategy to reduce sensor-to-edge transmission.

## Files

```text
data_generator.py              # Generates synthetic IoT data
dataset.py                     # Builds PyTorch datasets
transformer_fusion_model.py    # Defines the Transformer model
train.py                       # Trains and evaluates the model
```

## Usage

Generate the dataset:

```bash
python data_generator.py
```

Train and evaluate the model:

```bash
python train.py
```
