package encoder;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * A simple implementation as base for most REST encoders.
 */
public abstract class SimpleRESTEncoder extends encoder.AbstractRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(SimpleRESTEncoder.class);

  private Class<? extends Span> elementClass;
  private String vectorIdentifier;

  protected SimpleRESTEncoder(String id) {
    super(id);
  }
  
  public SimpleRESTEncoder(String id, encoder.RESTAdapter restAdapter, Class<? extends Span> elementClass) {
    super(id, restAdapter);
    this.elementClass = elementClass;
    this.vectorIdentifier = vectorIdentifier;
  }

  public SimpleRESTEncoder(String id, encoder.RESTAdapter restAdapter, String vectorIdentifier, Class<? extends Span> elementClass) {
    super(id, restAdapter);
    this.elementClass = elementClass;
    this.vectorIdentifier = vectorIdentifier;
  }

  @Override
  public INDArray encode(String word) {
    try{
      return encodeImpl(word);
    }catch (IOException e){
      log.error("IO Error while encoding word: {}", word, e);
      throw new UncheckedIOException(e);
    }
  }

  public abstract INDArray encodeImpl(String word) throws IOException;

  @Override
  public INDArray encode(Span span) {
    if(elementClass.isInstance(span)){
      try{
        return encodeImpl(span);
      }catch (IOException e){
        log.error("IO Error while encoding span: {}", span, e);
        throw new UncheckedIOException(e);
      }
    }else{
      throw new UnsupportedOperationException();
    }
  }

  public abstract INDArray encodeImpl(Span span) throws IOException;

  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    if(elementClass == this.elementClass){
      try {
        encodeEachImpl(input);
      }catch (IOException e){
        log.error("IO Error while encoding sentence: {}", input, e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException();
    }
  }

  public abstract void encodeEachImpl(Sentence input) throws IOException;

  @Override
  public void encodeEach(Document input, Class<? extends Span> elementClass) {
    if(elementClass == this.elementClass){
      try {
        encodeEachImpl(input);
      }catch (IOException e){
        log.error("IO Error while encoding document: {}", input.getTitle(), e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException();
    }
  }

  public abstract void encodeEachImpl(Document input) throws IOException;

  @Override
  public void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass) {
    if(elementClass == this.elementClass){
      try {
        encodeEachImpl(docs);
      }catch (IOException e){
        log.error("IO Error while encoding documents", e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException();
    }
  }

  public abstract void encodeEachImpl(Collection<Document> docs) throws IOException;
  
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
