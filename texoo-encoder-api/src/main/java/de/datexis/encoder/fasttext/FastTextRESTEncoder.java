package de.datexis.encoder.fasttext;

import de.datexis.encoder.RESTAdapter;
import de.datexis.encoder.SimpleRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

public class FastTextRESTEncoder extends SimpleRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(FastTextRESTEncoder.class);

  public static FastTextRESTEncoder create(String domain, int port) {
    return new FastTextRESTEncoder(new FastTextRESTAdapter(domain, port));
  }

  public static FastTextRESTEncoder create(String domain, int port, String vectorIdentifier) {
    return new FastTextRESTEncoder(new FastTextRESTAdapter(domain, port), vectorIdentifier);
  }

  public static FastTextRESTEncoder create(String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout) {
    return new FastTextRESTEncoder(new FastTextRESTAdapter(domain, port, embeddingVectorSize, connectTimeout, readTimeout));
  }

  public static FastTextRESTEncoder create(String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout, String vectorIdentifier) {
    return new FastTextRESTEncoder(new FastTextRESTAdapter(domain, port, embeddingVectorSize, connectTimeout, readTimeout), vectorIdentifier);
  }

  protected FastTextRESTEncoder() {
    super("FT");
  }
  
  public FastTextRESTEncoder(RESTAdapter restAdapter) {
    super("FT", restAdapter, Token.class);
  }

  public FastTextRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super("FT", restAdapter, vectorIdentifier, Token.class);
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
    for (Document document : docs) {
      encodeEachImpl(document);
    }
  }
}
