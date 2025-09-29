FROM python:3.9

ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=UTC

# Install basic apt packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    nano \
    vim \
    wget \
    curl

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install scikit-learn tqdm pyarrow numpy matplotlib pandas cython

RUN pip install scipy seaborn statsmodels

RUN pip install requests beautifulsoup4

# decision tree
RUN pip install lightgbm catboost xgboost optuna

# RUN pip install cleanlab imbalanced-learn

# symbolic regression
# RUN pip install pysr
# RUN python -c "from pysr import PySRRegressor; import numpy as np; \
#     X = np.random.randn(10, 3); y = np.random.randn(10); \
#     PySRRegressor(niterations=1).fit(X, y)"

ENTRYPOINT /bin/bash
