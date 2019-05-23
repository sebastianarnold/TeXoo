package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.encoder.impl.FastTextRESTAdapter;
import de.datexis.encoder.impl.FastTextRESTEncoder;
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

import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.*;

public class FastTextRESTEncoderTest {
    public static final String DUMMY_TEXT = "This is a sentence.";

    private List<Document> dummyDocuments;
    private Document dummyDocument;
    private Sentence dummySentence;

    private int vectorSize;

    private FastTextRESTAdapter fastTextRESTAdapter;
    private FastTextRESTEncoder fastTextRestEncoder;

    @Before
    public void setup() throws IOException {
        dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
        dummySentence = dummyDocument.getSentence(0);
        dummyDocuments = Lists.newArrayList(dummyDocument);

        fastTextRESTAdapter = mock(FastTextRESTAdapter.class);
        fastTextRestEncoder = spy(new FastTextRESTEncoder(fastTextRESTAdapter));

        vectorSize = (int) fastTextRestEncoder.getEmbeddingVectorSize();

        when(fastTextRESTAdapter.encode(Mockito.any(String[].class))).then(this::encodeTokenOfSentenceMock);
        when(fastTextRESTAdapter.encode(Mockito.any(String[][].class))).then(this::encodeTokenOfDocument2DMock);
    }

    private double[][] encodeTokenOfSentenceMock(InvocationOnMock invocationOnMock){
        String[] tokenOfSentence = invocationOnMock.getArgument(0);
        return new double[tokenOfSentence.length][vectorSize];
    }

    private double[][][] encodeTokenOfDocument2DMock(InvocationOnMock invocationOnMock){
        String[][] tokenOfDocument2D = invocationOnMock.getArgument(0);
        return new double[tokenOfDocument2D.length][tokenOfDocument2D[0].length][vectorSize];
    }

    @Test
    public void encodeEachTokenOfSentenceTest() throws IOException {
        fastTextRestEncoder.encodeEach(dummySentence, Token.class);

        verify(fastTextRestEncoder, times(1)).getTokensOfSentenceAsStringArray(eq(dummySentence));

        verify(fastTextRESTAdapter, times(1)).encode(Mockito.any(String[].class));

        verify(fastTextRestEncoder, times(1)).putVectorInTokenOfSentence(eq(dummySentence), Mockito.any(double[][].class));
    }

    @Test
    public void encodeEachTokenOfDocumentTest() throws IOException {
        fastTextRestEncoder.encodeEach(dummyDocument, Token.class);

        verify(fastTextRestEncoder, times(1)).getTokensOfDocumentAsStringArray2D(eq(dummyDocument));

        verify(fastTextRESTAdapter, times(1)).encode(Mockito.any(String[][].class));

        verify(fastTextRestEncoder, times(1)).putVectorInTokenOfDocument2D(eq(dummyDocument), Mockito.any(double[][][].class));
    }

    @Test
    public void encodeEachTokenOfDocumentsTest(){
        fastTextRestEncoder.encodeEach(dummyDocuments, Token.class);

        verify(fastTextRestEncoder, times(dummyDocuments.size())).encodeEach(Mockito.any(Document.class), eq(Token.class));
    }
}
