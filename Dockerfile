FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
            software-properties-common python-software-properties wget nano && \
    rm -rf /var/lib/apt/lists/*

# Install Java
RUN apt-get update && apt-get install openjdk-8-jdk -y

# Install maven 3.6.2
RUN wget --no-verbose -O /tmp/apache-maven-3.6.2-bin.tar.gz https://www-eu.apache.org/dist/maven/maven-3/3.6.2/binaries/apache-maven-3.6.2-bin.tar.gz && \
    tar xzf /tmp/apache-maven-3.6.2-bin.tar.gz -C /opt/ && \
    ln -s /opt/apache-maven-3.6.2 /opt/maven && \
    ln -s /opt/maven/bin/mvn /usr/local/bin  && \
    rm -f /tmp/apache-maven-3.6.2-bin.tar.gz

ENV MAVEN_HOME /opt/maven

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
RUN mvn -B -f /tmp/pom.xml verify --fail-never

# Define working directory
WORKDIR /usr/src

# Add TeXoo scripts to runtime path
ENV PATH /usr/src/bin:${PATH}

# Define default command.
CMD ["bash"]
