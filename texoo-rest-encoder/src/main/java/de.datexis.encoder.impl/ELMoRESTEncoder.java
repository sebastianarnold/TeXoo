package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

/**
 * Created by philipp on 02.10.18.
 */
public class ELMoRESTEncoder extends SimpleRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(ELMoRESTEncoder.class);

  public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput, String domain, int port) {
    return new ELMoRESTEncoder(new ELMoRESTAdapter(elMoLayerOutput, domain, port));
  }

  public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput, String domain, int port, String vectorIdentifier) {
    return new ELMoRESTEncoder(new ELMoRESTAdapter(elMoLayerOutput, domain, port), vectorIdentifier);
  }

  public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput, String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout) {
    return new ELMoRESTEncoder(new ELMoRESTAdapter(elMoLayerOutput, domain, port, embeddingVectorSize, connectTimeout, readTimeout));
  }

  public static ELMoRESTEncoder create(ELMoLayerOutput elMoLayerOutput, String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout, String vectorIdentifier) {
    return new ELMoRESTEncoder(new ELMoRESTAdapter(elMoLayerOutput, domain, port, embeddingVectorSize, connectTimeout, readTimeout), vectorIdentifier);
  }

  public ELMoRESTEncoder(RESTAdapter restAdapter) {
    super(restAdapter, Token.class);
  }

  public ELMoRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super(restAdapter, vectorIdentifier, Token.class);
  }

  @Override
  public INDArray encodeImpl(String word) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public INDArray encodeImpl(Span span) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void encodeEachImpl(Sentence input) throws IOException {
    encodeEach1D(input.getTokens());
  }

  @Override
  public void encodeEachImpl(Document input) throws IOException {
    encodeEach2D(getTokensOfSentencesOfDocument(input));
  }

  @Override
  public void encodeEachImpl(Collection<Document> docs) throws IOException {
    for(Document document: docs){
      encodeEachImpl(document);
    }
  }
}
