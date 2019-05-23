package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.encoder.impl.SkipthoughtRESTAdapter;
import de.datexis.encoder.impl.SkipthoughtRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.preprocess.DocumentFactory;
import org.junit.Before;
import org.junit.Test;
import org.mockito.Mockito;
import org.mockito.invocation.InvocationOnMock;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.List;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.contains;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

public class SkipthoughtRESTEncoderTest {
    public static final String DUMMY_TEXT = "This is a sentence.";
    public static final String DUMMY_SENTENCE = "This is a sentence.";

    private List<Document> dummyDocuments;
    private Document dummyDocument;
    private Sentence dummySentence;

    private long[] dummyShape;

    private int vectorSize;

    private SkipthoughtRESTAdapter skipthoughtRESTAdapter;
    private SkipthoughtRESTEncoder skipthoughtRESTEncoder;

    @Before
    public void setup() throws IOException {
        dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
        dummySentence = dummyDocument.getSentence(0);
        dummyDocuments = Lists.newArrayList(dummyDocument);

        skipthoughtRESTAdapter = mock(SkipthoughtRESTAdapter.class);
        skipthoughtRESTEncoder = spy(new SkipthoughtRESTEncoder(skipthoughtRESTAdapter));

        vectorSize = (int)skipthoughtRESTEncoder.getEmbeddingVectorSize();

        when(skipthoughtRESTAdapter.encode(anyString())).thenReturn(new double[vectorSize]);
        when(skipthoughtRESTAdapter.encode(Mockito.any(String[].class))).then(this::encodeSentenceOfDocumentMock);

        dummyShape = new long[]{vectorSize, 1};
    }

    private double[][] encodeSentenceOfDocumentMock(InvocationOnMock invocationOnMock){
        String[] sentencesOfDocument = invocationOnMock.getArgument(0);
        return new double[sentencesOfDocument.length][vectorSize];
    }

    @Test
    public void encodeSentenceTest() throws IOException {
        INDArray array = skipthoughtRESTEncoder.encode(dummySentence);

        long[] arrayShape = array.shape();

        assertThat(Lists.newArrayList(arrayShape), contains(dummyShape));

        verify(skipthoughtRESTEncoder, times(1)).encode(eq(DUMMY_SENTENCE));

        verify(skipthoughtRESTAdapter, times(1)).encode(eq(DUMMY_SENTENCE));
    }

    @Test
    public void encodeSentenceStringTest() throws IOException {
        INDArray array = skipthoughtRESTEncoder.encode(dummySentence.getText());

        long[] arrayShape = array.shape();

        assertThat(Lists.newArrayList(arrayShape), contains(dummyShape));

        verify(skipthoughtRESTAdapter, times(1)).encode(eq(DUMMY_SENTENCE));
    }

    @Test
    public void encodeEachSentenceTest() throws IOException {
        skipthoughtRESTEncoder.encodeEach(dummySentence, Sentence.class);

        verify(skipthoughtRESTAdapter, times(1)).encode(eq(DUMMY_SENTENCE));

        verify(skipthoughtRESTEncoder, times(1)).putVectorInSentence(eq(dummySentence), Mockito.any(double[].class));
    }

    @Test
    public void encodeEachSentenceInDocumentTest() throws IOException {
        skipthoughtRESTEncoder.encodeEach(dummyDocument, Sentence.class);

        verify(skipthoughtRESTEncoder, times(1)).getSentencesOfDocumentAsStringArray(eq(dummyDocument));

        verify(skipthoughtRESTAdapter, times(1)).encode(Mockito.any(String[].class));

        verify(skipthoughtRESTEncoder, times(1)).putVectorInSentenceOfDocument(eq(dummyDocument), Mockito.any(double[][].class));
    }

    @Test
    public void encodeEachSentenceInDocumentsTest(){
        skipthoughtRESTEncoder.encodeEach(dummyDocuments, Sentence.class);

        verify(skipthoughtRESTEncoder, times(dummyDocuments.size())).encodeEach(Mockito.any(Document.class), eq(Sentence.class));
    }
}
