package de.datexis.encoder.impl;

import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class AbstractRESTEncoder extends Encoder {
  private static final Logger log = LoggerFactory.getLogger(AbstractRESTEncoder.class);

  private RESTAdapter restAdapter;
  private String vectorIdentifier;

  public AbstractRESTEncoder(RESTAdapter restAdapter) {
    this(restAdapter, null);
  }

  public AbstractRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier) {
    this.restAdapter = restAdapter;
    this.vectorIdentifier = vectorIdentifier;
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

  public INDArray encodeValue(String value) throws IOException{
    return Nd4j.create(restAdapter.encode(value), new long[] {getEmbeddingVectorSize(), 1});
  }

  public List<List<Token>> getTokensOfSentencesOfDocument(Document document) {
    return document.streamSentences()
      .map(Sentence::getTokens)
      .collect(Collectors.toList());
  }

  public Stream<Stream<Token>> streamTokensOfSentencesOfDocument(Document document) {
    return document.streamSentences()
      .map(Sentence::streamTokens);
  }

  public <S> Stream<Stream<S>> streamSpans2D(List<? extends List<S>> spans2D){
    return spans2D.stream()
      .map(List::stream);
  }

  public <S extends Span> void encodeEach(S span) throws IOException {
    encodeEach(span, Span::getText);
  }

  public <S extends Span> void encodeEach(S span, Function<S, String> getText) throws IOException {
    String text = getText.apply(span);
    double[] embedding = restAdapter.encode(text);
    putVectorInSpan(span, embedding);
  }

  public <S extends Span> void encodeEach1D(List<S> spans) throws IOException {
    encodeEach1D(spans, Span::getText);
  }

  public <S extends Span> void encodeEach1D(List<S> spans, Function<S, String> getText) throws IOException {
    String[] spansAsStringArray1D = spansToStringArray1D(spans.stream(), getText);
    double[][] embedding = restAdapter.encode(spansAsStringArray1D);
    putVectorInSpans(spans.stream(), embedding);
  }

  public <S extends Span> void encodeEach2D(List<? extends List<S>> spans) throws IOException {
    encodeEach2D(spans, Span::getText);
  }

  public <S extends Span> void encodeEach2D(List<? extends List<S>> spans, Function<S, String> getText) throws IOException {
    String[][] spansAsStringArray2D = spansToStringArray2D(streamSpans2D(spans), getText);
    double[][][] embedding = restAdapter.encode(spansAsStringArray2D);
    putVectorInSpans(streamSpans2D(spans), embedding);
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
    if(vectorIdentifier == null){
      span.putVector(getClass(), Nd4j.create(data, new long[] {getEmbeddingVectorSize(), 1}));
    }else{
      span.putVector(vectorIdentifier, Nd4j.create(data, new long[] {getEmbeddingVectorSize(), 1}));
    }
  }

  public <S extends Span> void putVectorInSpans(Stream<S> spans, double[][] data){
    AtomicInteger i = new AtomicInteger();
    spans.forEach(span -> putVectorInSpan(span, data[i.getAndIncrement()]));
  }

  public <S extends Span> void putVectorInSpans(Stream<? extends Stream<S>> spans, double[][][] data){
    AtomicInteger i = new AtomicInteger();
    spans.forEach(span -> putVectorInSpans(span, data[i.getAndIncrement()]));
  }
}
