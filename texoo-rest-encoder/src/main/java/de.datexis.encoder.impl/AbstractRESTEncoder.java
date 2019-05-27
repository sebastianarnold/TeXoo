package de.datexis.encoder.impl;

import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Stream;

public abstract class AbstractRESTEncoder extends Encoder {
  private static final Logger log = LoggerFactory.getLogger(AbstractRESTEncoder.class);

  private RESTAdapter restAdapter;

  public AbstractRESTEncoder(RESTAdapter restAdapter) {
    this.restAdapter = restAdapter;
  }

  @Override
  public long getEmbeddingVectorSize() {
    return restAdapter.getEmbeddingVectorSize();
  }

  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("REST Encoders are not trainable");
  }

  @Override
  public void trainModel(Stream<Document> documents) {
    throw new UnsupportedOperationException("REST Encoders are not trainable");
  }

  @Override
  public void loadModel(Resource file) throws IOException {
    throw new UnsupportedOperationException("REST Encoders cant load a model");
  }

  @Override
  public void saveModel(Resource dir, String name) throws IOException {
    throw new UnsupportedOperationException("REST Encoders cant save a model");
  }

  public Stream<Stream<Token>> getTokensOfSentencesOfDocument(Document document) {
    return document.streamSentences()
      .map(Sentence::streamTokens);
  }

  public <S extends Span> void encodeEach(S span) throws IOException {
    encodeEach(span, Span::getText);
  }

  public <S extends Span> void encodeEach(S span, Function<S, String> getText) throws IOException {
    String text = getText.apply(span);
    double[] embedding = restAdapter.encode(text);
    putVectorInSpan(span, embedding);
  }

  public <S extends Span> void encodeEach1D(Stream<S> spans) throws IOException {
    encodeEach1D(spans, Span::getText);
  }

  public <S extends Span> void encodeEach1D(Stream<S> spans, Function<S, String> getText) throws IOException {
    String[] spansAsStringArray1D = spansToStringArray1D(spans, getText);
    double[][] embedding = restAdapter.encode(spansAsStringArray1D);
    putVectorInSpans(spans, embedding);
  }

  public <S extends Span> void encodeEach2D(Stream<? extends Stream<S>> spans) throws IOException {
    encodeEach2D(spans, Span::getText);
  }

  public <S extends Span> void encodeEach2D(Stream<? extends Stream<S>> spans, Function<S, String> getText) throws IOException {
    String[][] spansAsStringArray2D = spansToStringArray2D(spans, getText);
    double[][][] embedding = restAdapter.encode(spansAsStringArray2D);
    putVectorInSpans(spans, embedding);
  }

  public <S extends Span> String[] spansToStringArray1D(Stream<S> spans){
    return spansToStringArray1D(spans, Span::getText);
  }

  public <S extends Span> String[] spansToStringArray1D(Stream<S> spans, Function<S, String> getText){
    return spans
      .map(getText)
      .toArray(String[]::new);
  }

  public <S extends Span> String[][] spansToStringArray2D(Stream<? extends Stream<S>> spans){
    return spans
      .map(this::spansToStringArray1D)
      .toArray(String[][]::new);
  }

  public <S extends Span> String[][] spansToStringArray2D(Stream<? extends Stream<S>> spans, Function<S, String> getText){
    return spans
      .map(span -> this.spansToStringArray1D(span, getText))
      .toArray(String[][]::new);
  }

  public <S extends Span> void putVectorInSpan(S span, double[] data){
    span.putVector(getClass(), Nd4j.create(data, new long[] {getEmbeddingVectorSize(), 1}));
  }

  public <S extends Span> void putVectorInSpans(Stream<S> spans, double[][] data){
    AtomicInteger i = new AtomicInteger();
    spans.forEach(span -> putVectorInSpan(span, data[i.getAndIncrement()]));
  }

  public <S extends Span> void putVectorInSpans(Stream<? extends Stream<S>> spans, double[][][] data){
    AtomicInteger i = new AtomicInteger();
    spans.forEach(span -> putVectorInSpans(span, data[i.getAndIncrement()]));
  }

  /*public void putVectorInToken(Token token, double[] data) {
    token.putVector(getClass(), Nd4j.create(data, new long[] {getEmbeddingVectorSize(), 1}));
  }

  public void putVectorInSentence(Sentence sentence, double[] data) {
    sentence.putVector(getClass(), Nd4j.create(data, new long[] {getEmbeddingVectorSize(), 1}));
  }

  public void putVectorInTokenOfSentence(Sentence sentence, double[][] data) {
    int i = 0;
    for (Token token : sentence.getTokens()) {
      token.putVector(getClass(), Nd4j.create(data[i++], new long[] {getEmbeddingVectorSize(), 1}));
    }
  }

  public void putVectorInDocument(Document document, double[] data) {
    document.putVector(getClass(), Nd4j.create(data, new long[] {getEmbeddingVectorSize(), 1}));
  }

  public void putVectorInSentenceOfDocument(Document document, double[][] data) {
    int i = 0;
    for (Sentence sentence : document.getSentences()) {
      sentence.putVector(
          getClass(), Nd4j.create(data[i++], new long[] {getEmbeddingVectorSize(), 1}));
    }
  }

  public void putVectorInTokenOfDocument1D(Document document, double[][] data) {
    int i = 0;
    for (Token token : document.getTokens()) {
      token.putVector(getClass(), Nd4j.create(data[i++], new long[] {getEmbeddingVectorSize(), 1}));
    }
  }

  public void putVectorInTokenOfDocument2D(Document document, double[][][] data) {
    int i = 0;
    for (Sentence sentence : document.getSentences()) {
      int n = 0;
      for (Token token : sentence.getTokens()) {
        token.putVector(
            getClass(), Nd4j.create(data[i][n++], new long[] {getEmbeddingVectorSize(), 1}));
      }
      i++;
    }
  }*/
}
