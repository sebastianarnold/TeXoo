package de.datexis.encoder.impl;

import com.google.common.collect.Lists;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.preprocess.DocumentFactory;
import org.hamcrest.Matcher;
import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.equalTo;
import static org.mockito.Mockito.spy;

public class AbstractRESTEncoderTest {
    public static final String DUMMY_TEXT = "This is a dummy text.";

    public static final String[] TOKENS_OF_SENTENCE = new String[]{
            "This",
            "is",
            "a",
            "dummy",
            "text",
            "."
    };

    public static final String[][] TOKENS_OF_DOCUMENT_2D = new String[][]{
            {
                    "This",
                    "is",
                    "a",
                    "dummy",
                    "text",
                    "."
            }

    };

    public static final String[] TOKENS_OF_DOCUMENT_1D = new String[]{
            "This",
            "is",
            "a",
            "dummy",
            "text",
            "."
    };

    public static final String[] SENTENCES_OF_DOCUMENT = new String[]{
            "This is a dummy text."
    };

    private Document dummyDocument;
    private Sentence dummySentence;
    private Token dummyToken;

    private double[] dummyVectorToken;
    private double[] dummyVectorSentence;
    private double[] dummyVectorDocument;
    private double[][] dummyVectorTokenOfSentence;
    private double[][] dummyVectorSentenceOfDocument;
    private double[][] dummyVectorTokenOfDocument1D;
    private double[][][] dummyVectorTokenOfDocument2D;

    private long[] dummyShape;

    private AbstractRESTEncoder abstractRESTEncoder;

    @Before
    public void setup(){
        dummyDocument = DocumentFactory.fromText(DUMMY_TEXT);
        dummySentence = dummyDocument.getSentence(0);
        dummyToken = dummySentence.getToken(0);

        abstractRESTEncoder = spy(new DummyAbstractRESTEncoder());

        dummyShape = new long[]{abstractRESTEncoder.getEmbeddingVectorSize(), 1};

        dummyVectorToken = new double[(int)abstractRESTEncoder.getEmbeddingVectorSize()];
        dummyVectorSentence = new double[(int)abstractRESTEncoder.getEmbeddingVectorSize()];
        dummyVectorDocument = new double[(int)abstractRESTEncoder.getEmbeddingVectorSize()];
        dummyVectorTokenOfSentence = new double[dummySentence.getTokens().size()][(int)abstractRESTEncoder.getEmbeddingVectorSize()];
        dummyVectorSentenceOfDocument = new double[dummyDocument.getSentences().size()][(int)abstractRESTEncoder.getEmbeddingVectorSize()];
        dummyVectorTokenOfDocument1D = new double[dummyDocument.getTokens().size()][(int)abstractRESTEncoder.getEmbeddingVectorSize()];
        dummyVectorTokenOfDocument2D = dummyDocument.streamSentences()
                .map(s -> new double[s.getTokens().size()][(int)abstractRESTEncoder.getEmbeddingVectorSize()])
                .toArray(double[][][]::new);
    }

    @Test
    public void getTokensOfSentenceAsStringArrayTest(){
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

        System.out.println(Stream.of(TOKENS_OF_DOCUMENT_2D).map(Arrays::toString).collect(Collectors.joining(",")));
        System.out.println(Stream.of(result)               .map(Arrays::toString).collect(Collectors.joining(",")));

        List<Matcher<Iterable<? extends String>>> sentenceMatchers = Stream.of(TOKENS_OF_DOCUMENT_2D)
                .map(Matchers::contains)
                .collect(Collectors.toList());

        List<List<String>> sentences = Stream.of(TOKENS_OF_DOCUMENT_2D)
                .map(Lists::newArrayList)
                .collect(Collectors.toList());

        assertThat(TOKENS_OF_DOCUMENT_2D, equalTo(TOKENS_OF_DOCUMENT_2D));
    }

    @Test
    public void getSentenceIfDocumentAsStringArrayTest() {
        String[] result = abstractRESTEncoder.getSentencesOfDocumentAsStringArray(dummyDocument);

        assertThat(result, equalTo(SENTENCES_OF_DOCUMENT));
    }

    @Test
    public void putVectorInTokenTest(){
        abstractRESTEncoder.putVectorInToken(dummyToken, dummyVectorToken);

        long[] resultShape = dummyToken.getVector(abstractRESTEncoder.getClass()).shape();

        assertThat(resultShape, equalTo(dummyShape));
    }

    @Test
    public void putVectorInSentenceTest(){
        abstractRESTEncoder.putVectorInSentence(dummySentence, dummyVectorSentence);

        long[] resultShape = dummySentence.getVector(abstractRESTEncoder.getClass()).shape();

        assertThat(resultShape, equalTo(dummyShape));
    }

    @Test
    public void putVectorInDocumentTest(){
        abstractRESTEncoder.putVectorInDocument(dummyDocument, dummyVectorDocument);

        long[] resultShape = dummyDocument.getVector(abstractRESTEncoder.getClass()).shape();

        assertThat(resultShape, equalTo(dummyShape));
    }

    @Test
    public void putVectorInTokenOfSentenceTest(){
        abstractRESTEncoder.putVectorInTokenOfSentence(dummySentence, dummyVectorTokenOfSentence);

        for(Token token: dummySentence.getTokens()){
            long[] resultShape = token.getVector(abstractRESTEncoder.getClass()).shape();

            assertThat(resultShape, equalTo(dummyShape));
        }
    }

    @Test
    public void putVectorInSentenceOfDocumentTest(){
        abstractRESTEncoder.putVectorInSentenceOfDocument(dummyDocument, dummyVectorSentenceOfDocument);

        for(Sentence sentence: dummyDocument.getSentences()){
            long[] resultShape = sentence.getVector(abstractRESTEncoder.getClass()).shape();

            assertThat(resultShape, equalTo(dummyShape));
        }
    }

    @Test
    public void putVectorInTokenOfDocuemnt1DTest(){
        abstractRESTEncoder.putVectorInTokenOfDocument1D(dummyDocument, dummyVectorTokenOfDocument1D);

        for(Token token: dummyDocument.getTokens()){
            long[] resultShape = token.getVector(abstractRESTEncoder.getClass()).shape();

            assertThat(resultShape, equalTo(dummyShape));
        }
    }

    @Test
    public void putVectorInTokenOfDocument2DTest(){
        abstractRESTEncoder.putVectorInTokenOfDocument2D(dummyDocument, dummyVectorTokenOfDocument2D);

        for(Sentence sentence: dummyDocument.getSentences()){
            for(Token token: sentence.getTokens()){
                long[] resultShape = token.getVector(abstractRESTEncoder.getClass()).shape();

                assertThat(resultShape, equalTo(dummyShape));
            }
        }
    }
}
