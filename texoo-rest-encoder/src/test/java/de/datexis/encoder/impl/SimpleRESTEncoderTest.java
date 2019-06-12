package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.List;

import static org.mockito.Mockito.*;

public class SimpleRESTEncoderTest {
  public static final int EMBEDDING_VECTOR_SIZE = 100;
  public static final String DUMMY_TEXT = "This is a dummy text.";

  public Document dummyDocument;
  public Sentence dummySentence;
  public Token dummyToken;

  public RESTAdapter restAdapter;
  public SimpleRESTEncoder simpleRESTEncoder;

  @Before
  public void setup(){
    dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
    dummySentence = dummyDocument.getSentence(0);
    dummyToken = dummySentence.getToken(0);

    restAdapter = spy(new DummyRESTAdapter(EMBEDDING_VECTOR_SIZE));

    simpleRESTEncoder = spy(new DummySimpleRESTEncoder(restAdapter, Token.class));
  }

  @Test
  public void encodeTest() throws IOException {
    simpleRESTEncoder.encode(DUMMY_TEXT);

    verify(simpleRESTEncoder, times(1)).encodeImpl(eq(DUMMY_TEXT));
  }

  @Test(expected = UnsupportedOperationException.class)
  public void encodeSentenceTest(){
    simpleRESTEncoder.encode(dummySentence);
  }

  @Test
  public void encodeTokenTest() throws IOException{
    simpleRESTEncoder.encode(dummyToken);

    verify(simpleRESTEncoder, times(1)).encodeImpl(eq(dummyToken));
  }

  @Test(expected = UnsupportedOperationException.class)
  public void encodeEachSentenceSentenceTest(){
    simpleRESTEncoder.encodeEach(dummySentence, Sentence.class);
  }

  @Test
  public void encodeEachSentenceTokenTest() throws IOException{
    simpleRESTEncoder.encodeEach(dummySentence, Token.class);

    verify(simpleRESTEncoder, times(1)).encodeEachImpl(eq(dummySentence));
  }

  @Test(expected = UnsupportedOperationException.class)
  public void encodeEachDocumentSentenceTest(){
    simpleRESTEncoder.encodeEach(dummyDocument, Sentence.class);
  }

  @Test
  public void encodeEachDocumentTokenTest() throws IOException{
    simpleRESTEncoder.encodeEach(dummyDocument, Token.class);

    verify(simpleRESTEncoder, times(1)).encodeEachImpl(eq(dummyDocument));
  }

  @Test(expected = UnsupportedOperationException.class)
  public void encodeEachDocumentsSentenceTest(){
    simpleRESTEncoder.encodeEach(Lists.newArrayList(dummyDocument), Sentence.class);
  }

  @Test
  public void encodeEachDocumentsTokenTest() throws IOException{
    List<Document> docs = Lists.newArrayList(dummyDocument);

    simpleRESTEncoder.encodeEach(docs, Token.class);

    verify(simpleRESTEncoder, times(1)).encodeEachImpl(eq(docs));
  }
}
