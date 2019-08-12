package de.datexis.encoder.skipthought;

import de.datexis.encoder.RESTAdapter;
import de.datexis.encoder.SimpleRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

public class SkipthoughtRESTEncoder extends SimpleRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(SkipthoughtRESTEncoder.class);

  public static SkipthoughtRESTEncoder create(String domain, int port) {
    return new SkipthoughtRESTEncoder(new SkipthoughtRESTAdapter(domain, port));
  }

  public static SkipthoughtRESTEncoder create(String domain, int port, String vectorIdentifier) {
    return new SkipthoughtRESTEncoder(new SkipthoughtRESTAdapter(domain, port), vectorIdentifier);
  }

  public static SkipthoughtRESTEncoder create(String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout) {
    return new SkipthoughtRESTEncoder(new SkipthoughtRESTAdapter(domain, port, embeddingVectorSize, connectTimeout, readTimeout));
  }

  public static SkipthoughtRESTEncoder create(String domain, int port, long embeddingVectorSize, int connectTimeout, int readTimeout, String vectorIdentifier) {
    return new SkipthoughtRESTEncoder(new SkipthoughtRESTAdapter(domain, port, embeddingVectorSize, connectTimeout, readTimeout), vectorIdentifier);
  }

  protected SkipthoughtRESTEncoder() {
    super("ST");
  }
  
  public SkipthoughtRESTEncoder(RESTAdapter restAdapter) {
    super("ST", restAdapter, Sentence.class);
  }

  public SkipthoughtRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super("ST", restAdapter, vectorIdentifier, Sentence.class);
  }

  @Override
  public INDArray encodeImpl(String word) throws IOException {
    return encodeValue(word);
  }

  @Override
  public INDArray encodeImpl(Span span) throws IOException {
    return encodeImpl(span.getText());
  }

  @Override
  public void encodeEachImpl(Sentence input) throws IOException {
    encodeEach(input);
  }

  @Override
  public void encodeEachImpl(Document input) throws IOException {
    encodeEach1D(input.getSentences());
  }

  @Override
  public void encodeEachImpl(Collection<Document> docs) throws IOException {
    for (Document document : docs) {
      encodeEachImpl(document);
    }
  }
}
