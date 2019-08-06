package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.encoder.impl.BERTAsAService.BertRESTAdapter;
import de.datexis.encoder.impl.BERTAsAService.BertRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.List;


/*
 Integration-Test for the BertAsAService-REST-Encoder-Client
 The endpoint needs to be running at the defined address with the following parameters
 -pooling_strategy NONE
 Like this we get an embedding for every token
 */

public class BertRESTEncoderTest {
  public static final int EMBEDDING_VECTOR_SIZE = 1024;
  public static final String domain = "localhost";
  public static final int port = 8125;


  private final String DUMMY_TEXT = "This is a sentence. And this is another Sentence!";

  private List<Document> dummyDocuments;
  private Document dummyDocument;
  private Sentence dummySentence;

  private BertRESTAdapter restAdapter;
  private BertRESTEncoder bertRESTEncoder;

  @Before
  public void setup() throws IOException {
    dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
    dummySentence = dummyDocument.getSentence(0);
    dummyDocuments = Lists.newArrayList(dummyDocument);

    restAdapter = new BertRESTAdapter("localhost", 8125, EMBEDDING_VECTOR_SIZE);
    bertRESTEncoder = new BertRESTEncoder(restAdapter, "testIdentifier");
  }


  @Test
  public void encodeEachImplTokenOfSentenceTest() throws IOException {
    bertRESTEncoder.encodeEachImpl(dummySentence);
    for (Token t : dummySentence.getTokens()) {
      INDArray embedding = t.getVector("testIdentifier");
      Assert.assertNotNull(embedding);
      Assert.assertTrue(embedding.toFloatVector().length == 1024);
    }
  }

  @Test
  public void encodeEachImplTokenOfDocumentTest() throws IOException {
    bertRESTEncoder.encodeEachImpl(dummyDocument);
    for (Sentence s : dummyDocument.getSentences()) {
      for (Token t : s.getTokens()) {
        INDArray embedding = t.getVector("testIdentifier");
        Assert.assertNotNull(embedding);
        Assert.assertTrue(embedding.toFloatVector().length == 1024);
      }
    }
  }

  @Test
  public void encode() throws IOException {
    bertRESTEncoder.encodeEachImpl(dummySentence);
    for (Token t : dummySentence.getTokens()) {
      INDArray embedding = t.getVector("testIdentifier");
      Assert.assertNotNull(embedding);
      Assert.assertTrue(embedding.toFloatVector().length == 1024);
    }
  }

}
