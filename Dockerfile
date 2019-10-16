FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
            openjdk-8-jdk-headless wget nano

RUN apt-get install -y --no-install-recommends \
	    maven && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda/
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:$LD_LIBRARY_PATH
ENV OMP_NUM_THREADS=4
ENV CUDA_VISIBLE_DEVICES=0
ENV BACKEND_PRIORITY_CPU=1
ENV BACKEND_PRIORITY_GPU=2
ENV TEXOO_CUDA=10.1
ENV JAVA_TOOL_OPTIONS=-Dfile.encoding="UTF-8"
ENV MAVEN_OPTS="-XX:+UseG1GC -Xms4G -Xmx16G -Dorg.bytedeco.javacpp.maxbytes=16G -Dorg.bytedeco.javacpp.maxphysicalbytes=32G -Djavacpp.platform=linux-x86_64 -Dfile.encoding=UTF-8"

# Resolve dependencies at install time
COPY pom.xml /tmp/pom.xml
COPY texoo-cuda-9.2/pom.xml /tmp/texoo-cuda-9.2/pom.xml
COPY texoo-cuda-10.1/pom.xml /tmp/texoo-cuda-10.1/pom.xml
COPY texoo-core/pom.xml /tmp/texoo-core/pom.xml
COPY texoo-examples/pom.xml /tmp/texoo-examples/pom.xml
COPY texoo-retrieval/pom.xml /tmp/texoo-retrieval/pom.xml
COPY texoo-entity-recognition/pom.xml /tmp/texoo-entity-recognition/pom.xml
COPY texoo-entity-linking/pom.xml /tmp/texoo-entity-linking/pom.xml
COPY texoo-sector/pom.xml /tmp/texoo-sector/pom.xml
COPY texoo-encoder-api/pom.xml /tmp/texoo-encoder-api/pom.xml
COPY texoo-core/src/main/resources/texoo.properties.template /tmp/texoo-core/src/main/resources/texoo.properties.template
COPY texoo-core/build.xml /tmp/texoo-core/build.xml
RUN mvn -B -f /tmp/pom.xml verify -Djavacpp.platform=linux-x86_64 -DskipTests=true --fail-never

# Define working directory
WORKDIR /usr/src

# Add TeXoo scripts to runtime path
ENV PATH /usr/src/bin:${PATH}

# Define default command.
CMD ["bash"]
