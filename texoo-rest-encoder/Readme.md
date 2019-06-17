# Texoo - Rest Encoder

This module contains Encoders wich are useing an remote Embedding REST API. Therefore it is possible to run models which are only implemented in Python in the Texoo framework thought HTTPs. It is also possible to complex shift the calculation of an NN-Model in an distributed System. 

## Abstract:

This module consist out of classes with two different nameing shemes:

- \<Name\>RESTEncoder
- \<Name\>RESTAdapter

The RESTAdapters are responsible for makeing request at an specific API. The RESTEncoder on the other hand is responsible for implementing the Methods of an Texoo Encoder, preparing Texoo-data for transfer and linking the result data from the API to Texoo-data.

## Simple usage:

This module curently supports 4 Embedding APIs:

- [ELMo REST API](https://github.com/SchmaR/ELMo-Rest)
- [FastText REST API](https://github.com/devfoo-one/fastText-Rest)
- [Sector REST API](https://github.com/SchmaR/sector-encoder-rest)
- [Skipthought REST API](https://github.com/SchmaR/skipthought-rest)

In the following sections I will go over the steps required to get up and running with each Embedding API.

### ELMo REST API:

To get an setup ELMoRESTEncoder instance you should use folowing static functions in ELMoRESTEncoder:

```
public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput,
    String domain, int port)

public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput, 
    String domain, int port, String vectorIdentifier)

public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput, 
    String domain, int port, long embeddingVectorSize,
    int connectTimeout, int readTimeout)

public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput,
    String domain, int port, long embeddingVectorSize, 
    int connectTimeout, int readTimeout, String vectorIdentifier)
```

- elMoLayerOutput specifies which layer output of ELMo you want to use (top, middle, bottom, average)
- domain specifies the hostname or ip of the machine wich the API is runing on
- port specifies the port of the API
- embeddingVectorSize specifies the size of the vector the ELMo model of the API is returning
- connectTimeout specifies the connect timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setConnectTimeout-int-))
- readTimeout specifies the read timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setReadTimeout-int-))
- vectorIdentifier is used if you want that a embedding vector is linked via an name insted of the class of the Encoder.

### FastText REST API:

To get an setup FastTextRESTEncoder instance you should use folowing static functions in FastTextRESTEncoder:

```
public static FastTextRESTEncoder create(String domain, int port)

public static FastTextRESTEncoder create(String domain, int port, 
    String vectorIdentifier)

public static FastTextRESTEncoder create(String domain, int port, 
    long embeddingVectorSize, int connectTimeout, int readTimeout)

public static FastTextRESTEncoder create(String domain, int port, 
    long embeddingVectorSize, int connectTimeout, int readTimeout,
    String vectorIdentifier) 
```

- domain specifies the hostname or ip of the machine wich the API is runing on
- port specifies the port of the API
- embeddingVectorSize specifies the size of the vector the ELMo model of the API is returning
- connectTimeout specifies the connect timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setConnectTimeout-int-))
- readTimeout specifies the read timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setReadTimeout-int-))
- vectorIdentifier is used if you want that a embedding vector is linked via an name insted of the class of the Encoder.

### Sector REST API:

To get an setup ELMoRESTEncoder instance you should use folowing static functions in ELMoRESTEncoder:

```
public static SectorRESTEncoder create(String domain, int port)

public static SectorRESTEncoder create(String domain, int port, 
    String vectorIdentifier)

public static SectorRESTEncoder create(String domain, int port, 
    long embeddingVectorSize, int connectTimeout, int readTimeout) 

public static SectorRESTEncoder create(String domain, int port, 
    long embeddingVectorSize, int connectTimeout, int readTimeout, 
    String vectorIdentifier)
```

- domain specifies the hostname or ip of the machine wich the API is runing on
- port specifies the port of the API
- embeddingVectorSize specifies the size of the vector the ELMo model of the API is returning
- connectTimeout specifies the connect timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setConnectTimeout-int-))
- readTimeout specifies the read timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setReadTimeout-int-))
- vectorIdentifier is used if you want that a embedding vector is linked via an name insted of the class of the Encoder.

### Skipthought REST API:

To get an setup ELMoRESTEncoder instance you should use folowing static functions in ELMoRESTEncoder:

```
public static SkipthoughtRESTEncoder create(String domain, int port)

public static SkipthoughtRESTEncoder create(String domain, int port, 
    String vectorIdentifier)

public static SkipthoughtRESTEncoder create(String domain, int port, 
    long embeddingVectorSize, int connectTimeout, int readTimeout)

public static SkipthoughtRESTEncoder create(String domain, int port, 
    long embeddingVectorSize, int connectTimeout, int readTimeout, 
    String vectorIdentifier)
```

- domain specifies the hostname or ip of the machine wich the API is runing on
- port specifies the port of the API
- embeddingVectorSize specifies the size of the vector the ELMo model of the API is returning
- connectTimeout specifies the connect timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setConnectTimeout-int-))
- readTimeout specifies the read timeout of a request send. (See [This](https://docs.oracle.com/javase/8/docs/api/java/net/URLConnection.html#setReadTimeout-int-))
- vectorIdentifier is used if you want that a embedding vector is linked via an name insted of the class of the Encoder.

## Fundermetal Java Classes:

These Java Classes are required to Connect your Application to an specific Embedding REST API. 

| Package / Class | Description / Reference |
| --------------- | ----------------------- | 
| [ELMoRESTEncoder](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/ELMoRESTEncoder.java)    | ELMo REST Encoder |
| [ELMoRESTAdapter](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/ELMoRESTAdapter.java)   | ELMo REST Adapter |
| [ELMoLayerOutput](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/ELMoLayerOutput.java)    | Specifies the ELMo layer output you want use |
| [FastTextRESTEncoder](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/FastTextRESTEncoder.java) | FastText REST Encoder |
| [FastTextRESTAdapter](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/FastTextRESTAdapter.java) | FastText REST Adapter |
| [SectorRESTEncoder](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/SectorRESTEncoder.java) | Sector REST Encoder |
| [SectorRESTAdapter](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/SectorRESTAdapter.java) | Sector REST Adapter |
| [SkipthoughtRESTEncoder](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/SkipthoughtRESTEncoder.java) | Skipthought REST Encoder |
| [SkipthoughtRESTAdapter](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/SkipthoughtRESTAdapter.java) | Skipthought REST Adapter |

## Tips for connecting your own Embedding API

For inspiration start looking at ELMoRESTEncoder. If only have to encode only one type of span and you do not want to deal with IOExceptions I sugest using SimpleRESTEncoder.  

## Other Java Classes

These Java Classes are required to extend this module.

| Package / Class | Description / Reference |
| --------------- | ----------------------- | 
| [AbstractRESTEncoder](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/AbstractRESTEncoder.java) | Basis Class of all REST Encoders  |
| [AbstractRESTAdapter](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/AbstractRESTAdapter.java) | Basis Class of all REST Encoders |
| [SimpleRESTEncoder](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/SimpleRESTEncoder.java) | Specialized REST Encoder which only supports the encoding of one special Span type and deal with IOExceptions while encodeing |
| [DeserializationProvider](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/serde/DeserializationProvider.java) | An Interface for providing Deserialization Stategies |
| [SerializationProvider](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/serde/SerializationProvider.java) | An Interface for provideing Serialization Strategies |
| [JacksonSerdeProvider](texoo-rest-encoder/src/main/java/de/datexis/encoder/impl/serde/JacksonSerdeProvider.java) | An Serialization / Deserialization Stategy using Jackson  |
