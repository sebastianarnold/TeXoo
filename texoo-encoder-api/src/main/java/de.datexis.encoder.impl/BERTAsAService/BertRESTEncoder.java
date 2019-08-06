package de.datexis.encoder.impl.BERTAsAService;

import de.datexis.encoder.impl.RESTAdapter;
import de.datexis.encoder.impl.SimpleRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.Collection;

public class BertRESTEncoder extends SimpleRESTEncoder {

  public BertRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    super(restAdapter, vectorIdentifier, Token.class);
  }

  public BertRESTEncoder(RESTAdapter restAdapter) {
    super(restAdapter, Token.class);
  }

  public static BertRESTEncoder create(String domain, int port, String vectorIdentifier, int embeddingDimension, int connectionTimeout, int readTimeout) {
    return new BertRESTEncoder(new BertRESTAdapter(domain, port, embeddingDimension), vectorIdentifier);
  }


  @Override
  public INDArray encodeImpl(String word) throws IOException {
    return encode(word);
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
