 domains/fraud_detection/models/
├── __init__.py
├── algorithms/             # Model definitions, architectures, & custom neural nets
│   ├── __init__.py
│   ├── xgboost_wrapper.py  # Standardized XGBoost model wrapper
│   └── pytorch_net.py      # Custom Deep Learning architectures (PyTorch / TensorFlow)
│
├── lifecycle/              # Stage-by-stage model lifecycle execution logic
│   ├── train.py            # Training execution logic, loss tracking, & optimizer setup
│   ├── evaluate.py         # Domain-specific evaluation, thresholding, & slicing metrics
│   └── predict.py          # Batch & real-time inference prediction engines
│
├── factory.py              # Model Factory (Instantiates model objects based on config)
├── hyperparams.py          # Hyperparameter tuning logic (Optuna, Ray Tune setup)
├── exporter.py             # Packaging, serialization, & export (ONNX, TorchScript, MLflow)
└── registry.py             # Model Registry interface (MLflow / W&B client integration)
