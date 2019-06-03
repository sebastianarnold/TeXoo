package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.encoder.impl.ELMoRESTAdapter;
import de.datexis.encoder.impl.ELMoRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;

import java.io.IOException;
import java.util.List;
import java.util.stream.Stream;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

public class ELMoRESTEncoderTest {
  public static final int EMBEDDING_VECTOR_SIZE = 100;

  private final String DUMMY_TEXT = "This is a sentence.";

  private List<Document> dummyDocuments;
  private Document dummyDocument;
  private Sentence dummySentence;

  private RESTAdapter restAdapter;
  private ELMoRESTEncoder elMoRESTEncoder;

  @Before
  public void setup() throws IOException {
    dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
    dummySentence = dummyDocument.getSentence(0);
    dummyDocuments = Lists.newArrayList(dummyDocument);

    restAdapter = spy(new DummyRESTAdapter(EMBEDDING_VECTOR_SIZE));
    elMoRESTEncoder = spy(new ELMoRESTEncoder(restAdapter));

    // when(elMoRESTAdapter.encode(Mockito.any(String[].class))).then(this::encodeTokenOfSentenceMock);
    // when(elMoRESTAdapter.encode(Mockito.any(String[][].class))).then(this::encodeTokenOfDocument2DMock);
  }

  /*private double[][] encodeTokenOfSentenceMock(InvocationOnMock invocationOnMock) {
    String[] tokenOfSentence = invocationOnMock.getArgument(0);
    return new double[tokenOfSentence.length][vectorSize];
  }

  private double[][][] encodeTokenOfDocument2DMock(InvocationOnMock invocationOnMock) {
    String[][] tokenOfDocument2D = invocationOnMock.getArgument(0);
    return new double[tokenOfDocument2D.length][tokenOfDocument2D[0].length][vectorSize];
  }*/

  @Test(expected = UnsupportedOperationException.class)
  public void encodeImplTest(){
    elMoRESTEncoder.encode(DUMMY_TEXT);
  }

  @Test(expected = UnsupportedOperationException.class)
  public void encodeImplSentenceTest(){
    elMoRESTEncoder.encode(dummySentence);
  }

  @Test
  public void encodeEachImplTokenOfSentenceTest() throws IOException {
    elMoRESTEncoder.encodeEachImpl(dummySentence);

    verify(elMoRESTEncoder, times(1)).encodeEach1D(any());
  }

  @Test
  public void encodeEachImplTokenOfDocumentTest() throws IOException {
    elMoRESTEncoder.encodeEachImpl(dummyDocument);

    verify(elMoRESTEncoder, times(1)).getTokensOfSentencesOfDocument(eq(dummyDocument));

    verify(elMoRESTEncoder, times(1)).encodeEach2D(any());
  }

  @Test
  public void encodeEachImplTokenOfDocumentsTest() throws IOException {
    elMoRESTEncoder.encodeEachImpl(dummyDocuments);

    verify(elMoRESTEncoder, times(dummyDocuments.size()))
        .encodeEachImpl(Mockito.any(Document.class));
  }
}
