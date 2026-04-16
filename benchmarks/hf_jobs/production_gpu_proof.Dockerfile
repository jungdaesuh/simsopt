FROM python:3.11-bookworm

ARG SIMSOPT_HF_JOB_JAX_GPU_WHEEL_SPEC="jax[cuda12]==0.9.2"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    libboost-all-dev \
    libfftw3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    liblapack-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    libopenblas-dev \
    libopenmpi-dev \
    openmpi-bin \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      "numpy>=2.0" \
      cmake \
      scikit-build-core \
      ninja \
      "setuptools-scm>=8.0" \
      "scipy>=1.13" \
      pytest \
      sympy \
      f90nml \
      pyevtk \
      matplotlib \
      shapely \
      numba \
      "ground==9.0.0" \
      "bentley_ottmann==8.0.0" \
      ruamel.yaml \
      monty \
      Deprecated \
      "pybind11<3" && \
    python -m pip install "${SIMSOPT_HF_JOB_JAX_GPU_WHEEL_SPEC}"

WORKDIR /work

CMD ["bash"]
