package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.notNullValue;
import static org.mockito.Mockito.*;

public class AbstractRESTEncoderTest {
  public static final int EMBEDDING_VECTOR_SIZE = 100;
  public static final String VECTOR_IDENTIFIER = "AbstractRESTEncoder";
  public static final String DUMMY_TEXT = "This is a dummy text.";

  private Document dummyDocument;

  private DummySpan dummySpan;
  private List<DummySpan> dummySpan1D;
  private List<List<DummySpan>> dummySpan2D;

  private double[] dummyVectorSpan;
  private double[][] dummyVectorSpan1D;
  private double[][][] dummyVectorSpan2D;

  private long[] dummyShape;

  private AbstractRESTEncoder abstractRESTEncoder;
  private RESTAdapter restAdapter;

  @Before
  public void setup() throws IOException {
    dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
    /*dummySentence = dummyDocument.getSentence(0);
    dummyToken = dummySentence.getToken(0);*/

    dummySpan = new DummySpan();
    dummySpan1D = Lists.newArrayList(new DummySpan(), new DummySpan());
    dummySpan2D = Lists.newArrayList(
      Lists.newArrayList(new DummySpan(), new DummySpan()),
      Lists.newArrayList(new DummySpan(), new DummySpan())
    );

    restAdapter = spy(new DummyRESTAdapter(EMBEDDING_VECTOR_SIZE));

    abstractRESTEncoder = spy(new DummyAbstractRESTEncoder(restAdapter, VECTOR_IDENTIFIER));

    dummyShape = new long[] {EMBEDDING_VECTOR_SIZE, 1};

    dummyVectorSpan = new double[EMBEDDING_VECTOR_SIZE];
    dummyVectorSpan1D = new double[dummySpan1D.size()][EMBEDDING_VECTOR_SIZE];
    dummyVectorSpan2D = dummySpan2D.stream()
      .map(s -> new double[s.size()][EMBEDDING_VECTOR_SIZE])
      .toArray(double[][][]::new);

    /*dummyVectorToken = new double[EMBEDDING_VECTOR_SIZE];
    dummyVectorSentence = new double[EMBEDDING_VECTOR_SIZE];
    dummyVectorDocument = new double[EMBEDDING_VECTOR_SIZE];
    dummyVectorTokenOfSentence = new double[dummySentence.getTokens().size()][EMBEDDING_VECTOR_SIZE];
    dummyVectorSentenceOfDocument = new double[dummyDocument.getSentences().size()][EMBEDDING_VECTOR_SIZE];
    dummyVectorTokenOfDocument1D = new double[dummyDocument.getTokens().size()][EMBEDDING_VECTOR_SIZE];
    dummyVectorTokenOfDocument2D = dummyDocument.streamSentences()
            .map(s -> new double[s.getTokens().size()][EMBEDDING_VECTOR_SIZE])
            .toArray(double[][][]::new);*/
  }

  @Test
  public void getEmbeddingVectorSizeTest(){
    assertThat(abstractRESTEncoder.getEmbeddingVectorSize(), equalTo((long) EMBEDDING_VECTOR_SIZE));
  }

  @Test
  public void encodeImplTest() throws IOException {
    INDArray embedding = abstractRESTEncoder.encodeValue("text");

    assertThat(embedding.shape(), equalTo(dummyShape));
  }

  /*@Test
  public void streamTokensOfSentencesOfDocumentTest(){
    Stream<Stream<Token>> tokenStream2D = abstractRESTEncoder.streamTokensOfSentencesOfDocument(dummyDocument);

    assertThat(tokenStream2D.count(), equalTo((long)dummyDocument.countSentences()));

    AtomicInteger i = new AtomicInteger();
    tokenStream2D.forEach(tokenStream1D -> {
      Sentence sentence = dummyDocument.getSentence(i.getAndIncrement());

      assertThat(tokenStream1D.count(), equalTo((long)sentence.countTokens()));

      AtomicInteger n = new AtomicInteger();
      tokenStream1D.forEach(token -> {
        assertThat(token, equalTo(sentence.getToken(n.getAndIncrement())));
      });
    });
  }*/

  @Test
  public void getTokensOfSentencesOfDocumentTest(){
    List<List<Token>> tokens2D = abstractRESTEncoder.getTokensOfSentencesOfDocument(dummyDocument);

    assertThat(tokens2D.size(), equalTo(dummyDocument.countSentences()));

    for (int i = 0; i < tokens2D.size(); i++) {
      List<Token> tokens1D = tokens2D.get(i);
      Sentence sentence = dummyDocument.getSentence(i);

      assertThat(tokens1D.size(), equalTo(sentence.countTokens()));

      for (int n = 0; n < tokens1D.size(); n++){
        assertThat(tokens1D.get(n), equalTo(sentence.getToken(n)));
      }
    }
  }

  /*@Test
  public void streamSpans2DTest(){
    Stream<Stream<DummySpan>> spanStream2D = abstractRESTEncoder.streamSpans2D(dummySpan2D);

    assertThat(spanStream2D.count(), equalTo((long)dummySpan2D.size()));

    AtomicInteger i = new AtomicInteger();
    spanStream2D.forEach(spanStream1D -> {
      List<DummySpan> spans2D = dummySpan2D.get(i.getAndIncrement());

      assertThat(spanStream1D.count(), equalTo((long)spans2D.size()));

      AtomicInteger n = new AtomicInteger();
      spanStream1D.forEach(token -> {
        assertThat(token, equalTo(spans2D.get(n.getAndIncrement())));
      });
    });
  }*/

  @Test
  public void encodeEachTest() throws IOException {
    abstractRESTEncoder.encodeEach(dummySpan);

    verify(restAdapter, times(1)).encode(eq(DummySpan.TEXT));

    verifySpanVector(dummySpan);
  }

  @Test
  public void encodeEachCustomGetTextTest() throws IOException {
    abstractRESTEncoder.encodeEach(dummySpan, DummySpan::customGetText);

    verify(restAdapter, times(1)).encode(eq(DummySpan.CUSTOM_TEXT));

    verifySpanVector(dummySpan);
  }


  @Test
  public void encodeEach1DTest() throws IOException {
    abstractRESTEncoder.encodeEach1D(dummySpan1D);

    verify(restAdapter, times(1)).encode(Mockito.any(String[].class));

    for(Span span: dummySpan1D){
      verifySpanVector(span);
    }
  }

  @Test
  public void encodeEach1DCustomGetTextTest() throws IOException {
    abstractRESTEncoder.encodeEach1D(dummySpan1D, DummySpan::customGetText);

    verify(restAdapter, times(1)).encode(Mockito.any(String[].class));

    for(Span span: dummySpan1D){
      verifySpanVector(span);
    }
  }

  @Test
  public void encodeEach2DTest() throws IOException {
    abstractRESTEncoder.encodeEach2D(dummySpan2D);

    verify(restAdapter, times(1)).encode(Mockito.any(String[][].class));

    for(List<DummySpan> span1D: dummySpan2D){
      for(Span span : span1D){
        verifySpanVector(span);
      }
    }
  }

  @Test
  public void encodeEach2DCustomGetTextTest() throws IOException {
    abstractRESTEncoder.encodeEach2D(dummySpan2D, DummySpan::customGetText);

    verify(restAdapter, times(1)).encode(Mockito.any(String[][].class));

    for(List<DummySpan> span1D: dummySpan2D){
      for(Span span : span1D){
        verifySpanVector(span);
      }
    }
  }

  private void verifySpanVector(Span span){
    assertThat(span.getVector(VECTOR_IDENTIFIER), notNullValue());
    assertThat(span.getVector(VECTOR_IDENTIFIER).shape(), equalTo(dummyShape));
  }

  @Test
  public void spansToStringArray1DTest(){
    String[] spanString1D = abstractRESTEncoder.spansToStringArray1D(dummySpan1D.stream());

    for(String spanString: spanString1D){
      assertThat(spanString, equalTo(DummySpan.TEXT));
    }
  }

  @Test
  public void spansToStringArray1DCustomGetTextTest(){
    String[] spanString1D = abstractRESTEncoder.spansToStringArray1D(dummySpan1D.stream(), DummySpan::customGetText);

    for(String spanString: spanString1D){
      assertThat(spanString, equalTo(DummySpan.CUSTOM_TEXT));
    }
  }

  @Test
  public void spansToStringArray2DTest(){
    Stream<Stream<DummySpan>> spanStreams = dummySpan2D.stream()
      .map(List::stream);

    String[][] spanString2D = abstractRESTEncoder.spansToStringArray2D(spanStreams);

    for(String[] spanString1D: spanString2D){
      for(String spanString: spanString1D){
        assertThat(spanString, equalTo(DummySpan.TEXT));
      }
    }
  }

  @Test
  public void spansToStringArray2DCustomGetTextTest(){
    Stream<Stream<DummySpan>> spanStreams = dummySpan2D.stream()
      .map(List::stream);

    String[][] spanString2D = abstractRESTEncoder.spansToStringArray2D(spanStreams, DummySpan::customGetText);

    for(String[] spanString1D: spanString2D){
      for(String spanString: spanString1D){
        assertThat(spanString, equalTo(DummySpan.CUSTOM_TEXT));
      }
    }
  }

  @Test
  public void putVectorInSpanTest(){
    abstractRESTEncoder.putVectorInSpan(dummySpan, dummyVectorSpan);

    verifySpanVector(dummySpan);
  }

  @Test
  public void putVectorInSpans1D(){
    abstractRESTEncoder.putVectorInSpans(dummySpan1D.stream(), dummyVectorSpan1D);

    for(Span span: dummySpan1D){
      verifySpanVector(span);
    }
  }

  @Test
  public void putVectorInSpans2D(){
    Stream<Stream<DummySpan>> spanStreams = dummySpan2D.stream()
      .map(List::stream);

    abstractRESTEncoder.putVectorInSpans(spanStreams, dummyVectorSpan2D);

    for (List<DummySpan> span1D : dummySpan2D) {
      for (Span span : span1D) {
        verifySpanVector(span);
      }
    }
  }

  /*@Test
  public void getTokensOfSentenceAsStringArrayTest() {
    String[] result = abstractRESTEncoder.getTokensOfSentenceAsStringArray(dummySentence);

    assertThat(result, equalTo(TOKENS_OF_SENTENCE));
  }

  @Test
  public void getTokensOfDocumentAsStringArray1DTest() {
    String[] result = abstractRESTEncoder.getTokensOfDocumentAsStringArray1D(dummyDocument);

    assertThat(result, equalTo(TOKENS_OF_DOCUMENT_1D));
  }

  @Test
  public void getTokensOfDocumentAsStringArray2DTest() {
    String[][] result = abstractRESTEncoder.getTokensOfDocumentAsStringArray2D(dummyDocument);

    System.out.println(
        Stream.of(TOKENS_OF_DOCUMENT_2D).map(Arrays::toString).collect(Collectors.joining(",")));
    System.out.println(Stream.of(result).map(Arrays::toString).collect(Collectors.joining(",")));

    List<Matcher<Iterable<? extends String>>> sentenceMatchers =
        Stream.of(TOKENS_OF_DOCUMENT_2D).map(Matchers::contains).collect(Collectors.toList());

    List<List<String>> sentences =
        Stream.of(TOKENS_OF_DOCUMENT_2D).map(Lists::newArrayList).collect(Collectors.toList());

    assertThat(TOKENS_OF_DOCUMENT_2D, equalTo(TOKENS_OF_DOCUMENT_2D));
  }

  @Test
  public void getSentenceIfDocumentAsStringArrayTest() {
    String[] result = abstractRESTEncoder.getSentencesOfDocumentAsStringArray(dummyDocument);

    assertThat(result, equalTo(SENTENCES_OF_DOCUMENT));
  }

  @Test
  public void putVectorInTokenTest() {
    abstractRESTEncoder.putVectorInToken(dummyToken, dummyVectorToken);

    long[] resultShape = dummyToken.getVector(abstractRESTEncoder.getClass()).shape();

    assertThat(resultShape, equalTo(dummyShape));
  }

  @Test
  public void putVectorInSentenceTest() {
    abstractRESTEncoder.putVectorInSentence(dummySentence, dummyVectorSentence);

    long[] resultShape = dummySentence.getVector(abstractRESTEncoder.getClass()).shape();

    assertThat(resultShape, equalTo(dummyShape));
  }

  @Test
  public void putVectorInDocumentTest() {
    abstractRESTEncoder.putVectorInDocument(dummyDocument, dummyVectorDocument);

    long[] resultShape = dummyDocument.getVector(abstractRESTEncoder.getClass()).shape();

    assertThat(resultShape, equalTo(dummyShape));
  }

  @Test
  public void putVectorInTokenOfSentenceTest() {
    abstractRESTEncoder.putVectorInTokenOfSentence(dummySentence, dummyVectorTokenOfSentence);

    for (Token token : dummySentence.getTokens()) {
      long[] resultShape = token.getVector(abstractRESTEncoder.getClass()).shape();

      assertThat(resultShape, equalTo(dummyShape));
    }
  }

  @Test
  public void putVectorInSentenceOfDocumentTest() {
    abstractRESTEncoder.putVectorInSentenceOfDocument(dummyDocument, dummyVectorSentenceOfDocument);

    for (Sentence sentence : dummyDocument.getSentences()) {
      long[] resultShape = sentence.getVector(abstractRESTEncoder.getClass()).shape();

      assertThat(resultShape, equalTo(dummyShape));
    }
  }

  @Test
  public void putVectorInTokenOfDocuemnt1DTest() {
    abstractRESTEncoder.putVectorInTokenOfDocument1D(dummyDocument, dummyVectorTokenOfDocument1D);

    for (Token token : dummyDocument.getTokens()) {
      long[] resultShape = token.getVector(abstractRESTEncoder.getClass()).shape();

      assertThat(resultShape, equalTo(dummyShape));
    }
  }

  @Test
  public void putVectorInTokenOfDocument2DTest() {
    abstractRESTEncoder.putVectorInTokenOfDocument2D(dummyDocument, dummyVectorTokenOfDocument2D);

    for (Sentence sentence : dummyDocument.getSentences()) {
      for (Token token : sentence.getTokens()) {
        long[] resultShape = token.getVector(abstractRESTEncoder.getClass()).shape();

        assertThat(resultShape, equalTo(dummyShape));
      }
    }
  }*/
}
