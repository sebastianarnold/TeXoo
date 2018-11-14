FROM nvidia/cuda:10.0-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

# Install CuDNN and system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda10.0 openjdk-8-jdk-headless maven software-properties-common wget nano && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Resolve dependencies at install time
COPY pom.xml /tmp/pom.xml
COPY texoo-core/pom.xml /tmp/texoo-core/pom.xml
COPY texoo-examples/pom.xml /tmp/texoo-examples/pom.xml
COPY texoo-entity-recognition/pom.xml /tmp/texoo-entity-recognition/pom.xml
COPY texoo-entity-linking/pom.xml /tmp/texoo-entity-linking/pom.xml
COPY texoo-sector/pom.xml /tmp/texoo-sector/pom.xml
RUN mvn -B -f /tmp/pom.xml verify --fail-never

# Define working directory
WORKDIR /usr/src

# Add TeXoo scripts to runtime path
ENV PATH /usr/src/bin:${PATH}

# Define default command.
CMD ["bash"]
