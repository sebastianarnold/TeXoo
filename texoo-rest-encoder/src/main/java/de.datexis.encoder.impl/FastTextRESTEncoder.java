package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;

public class FastTextRESTEncoder extends AbstractRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(FastTextRESTEncoder.class);

  private FastTextRESTAdapter fastTextRestAdapter;

  public static FastTextRESTEncoder create(String domain, int port) {
    return new FastTextRESTEncoder(new FastTextRESTAdapter(domain, port));
  }

  public FastTextRESTEncoder(FastTextRESTAdapter fastTextRestAdapter) {
    this.fastTextRestAdapter = fastTextRestAdapter;
  }

  @Override
  public long getEmbeddingVectorSize() {
    return 300;
  }

  @Override
  public INDArray encode(String s) {
    throw new UnsupportedOperationException();
  }

  @Override
  public INDArray encode(Span span) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    if(elementClass == Token.class){
      try {
        String[] tokensOfSentenceAsStringArray = getTokensOfSentenceAsStringArray(input);

        double[][] embedding = fastTextRestAdapter.encode(tokensOfSentenceAsStringArray);

        putVectorInTokenOfSentence(input, embedding);
      } catch (IOException e) {
        log.error("IO Error while encoding sentence: {}", input, e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException("FastText can not encode anything else then Tokens");
    }
  }

  @Override
  public void encodeEach(Document input, Class<? extends Span> elementClass) {
    if(elementClass==Token.class){
      try {
        String[][] tokenOfDocumentAsStringArray2D = getTokensOfDocumentAsStringArray2D(input);

        double[][][] embedding = fastTextRestAdapter.encode(tokenOfDocumentAsStringArray2D);

        putVectorInTokenOfDocument2D(input, embedding);
      } catch (IOException e) {
        log.error("IO Error while encoding document: {}", input, e);
        throw new UncheckedIOException(e);
      }
    }else{
      throw new UnsupportedOperationException("FastText can not encode anything else then Tokens");
    }
  }

  @Override
  public void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass) {
    for (Document document : docs) {
      encodeEach(document, elementClass);
    }
  }

  @Override
  public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
    throw new UnsupportedOperationException();
  }
}
