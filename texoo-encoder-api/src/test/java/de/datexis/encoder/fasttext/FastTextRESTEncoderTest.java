package de.datexis.encoder.fasttext;

import com.google.common.collect.Lists;
import de.datexis.encoder.DummyRESTAdapter;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import encoder.fasttext.FastTextRESTEncoder;
import encoder.RESTAdapter;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;

import java.io.IOException;
import java.util.List;

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

public class FastTextRESTEncoderTest {
  public static final int EMBEDDING_VECTOR_SIZE = 100;

  public static final String DUMMY_TEXT = "This is a sentence.";

  private List<Document> dummyDocuments;
  private Document dummyDocument;
  private Sentence dummySentence;

  private RESTAdapter restAdapter;
  private FastTextRESTEncoder fastTextRestEncoder;

  @Before
  public void setup() throws IOException {
    dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
    dummySentence = dummyDocument.getSentence(0);
    dummyDocuments = Lists.newArrayList(dummyDocument);

    restAdapter = spy(new DummyRESTAdapter(EMBEDDING_VECTOR_SIZE));
    fastTextRestEncoder = spy(new FastTextRESTEncoder(restAdapter));
  }

  @Test(expected = UnsupportedOperationException.class)
  public void encodeImplTest(){
    fastTextRestEncoder.encode(DUMMY_TEXT);
  }

  @Test(expected = UnsupportedOperationException.class)
  public void encodeImplSentenceTest(){
    fastTextRestEncoder.encode(dummySentence);
  }

  @Test
  public void encodeEachImplTokenOfSentenceTest() throws IOException {
    fastTextRestEncoder.encodeEachImpl(dummySentence);

    verify(fastTextRestEncoder, times(1)).encodeEach1D(any());
  }

  @Test
  public void encodeEachImplTokenOfDocumentTest() throws IOException {
    fastTextRestEncoder.encodeEachImpl(dummyDocument);

    verify(fastTextRestEncoder, times(1)).encodeEach2D(any());

    verify(fastTextRestEncoder, times(1)).getTokensOfSentencesOfDocument(eq(dummyDocument));
  }

  @Test
  public void encodeEachImplTokenOfDocumentsTest() throws IOException {
    fastTextRestEncoder.encodeEachImpl(dummyDocuments);

    verify(fastTextRestEncoder, times(dummyDocuments.size()))
        .encodeEachImpl(Mockito.any(Document.class));
  }
}
