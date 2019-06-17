package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

public class SkipthoughtRESTEncoderTest {
  public static final int EMBEDDING_VECTOR_SIZE = 100;

  public static final String DUMMY_TEXT = "This is a sentence.";
  public static final String DUMMY_SENTENCE = "This is a sentence.";

  private List<Document> dummyDocuments;
  private Document dummyDocument;
  private Sentence dummySentence;

  private long[] dummyShape;

  private RESTAdapter restAdapter;
  private SkipthoughtRESTEncoder skipthoughtRESTEncoder;

  @Before
  public void setup() {
    dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
    dummySentence = dummyDocument.getSentence(0);
    dummyDocuments = Lists.newArrayList(dummyDocument);

    restAdapter = spy(new DummyRESTAdapter(EMBEDDING_VECTOR_SIZE));
    skipthoughtRESTEncoder = spy(new SkipthoughtRESTEncoder(restAdapter));

    dummyShape = new long[] {EMBEDDING_VECTOR_SIZE, 1};
  }

  @Test
  public void encodeImplSentenceTest() throws IOException {
    INDArray array = skipthoughtRESTEncoder.encode(dummySentence);

    assertThat(array.shape(), equalTo(dummyShape));

    verify(skipthoughtRESTEncoder, times(1)).encodeImpl(eq(DUMMY_SENTENCE));

    verify(skipthoughtRESTEncoder, times(1)).encodeValue(eq(DUMMY_SENTENCE));
  }

  @Test
  public void encodeImplSentenceStringTest() throws IOException {
    INDArray array = skipthoughtRESTEncoder.encode(DUMMY_SENTENCE);

    assertThat(array.shape(), equalTo(dummyShape));

    verify(skipthoughtRESTEncoder, times(1)).encodeValue(eq(DUMMY_SENTENCE));
  }

  @Test
  public void encodeEachImplSentenceTest() throws IOException {
    skipthoughtRESTEncoder.encodeEach(dummySentence, Sentence.class);

    verify(skipthoughtRESTEncoder, times(1)).encodeEach(eq(dummySentence));
  }

  @Test
  public void encodeEachImplSentenceInDocumentTest() throws IOException {
    skipthoughtRESTEncoder.encodeEach(dummyDocument, Sentence.class);

    verify(skipthoughtRESTEncoder, times(1)).encodeEach1D(any());
  }

  @Test
  public void encodeEachImplSentenceInDocumentsTest() throws IOException {
    skipthoughtRESTEncoder.encodeEach(dummyDocuments, Sentence.class);

    verify(skipthoughtRESTEncoder, times(dummyDocuments.size()))
        .encodeEachImpl(Mockito.any(Document.class));
  }
}
