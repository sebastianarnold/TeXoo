package de.datexis.encoder.impl;

import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public abstract class AbstractRESTEncoder extends Encoder {
    public void trainModel(Collection<Document> documents) {
        throw new UnsupportedOperationException("REST Encoders are not trainable");
    }

    public void trainModel(Stream<Document> documents) {
        trainModel(documents.collect(Collectors.toList()));
    }

    @Override
    public void loadModel(Resource file) throws IOException {
        throw new UnsupportedOperationException("REST Encoders cant load a model");
    }

    @Override
    public void saveModel(Resource dir, String name) throws IOException {
        throw new UnsupportedOperationException("REST Encoders cant save a model");
    }

    public String[] getTokensOfSentenceAsStringArray(Sentence sentence){
        return sentence.streamTokens()
                .map(Token::getText)
                .toArray(String[]::new);
    }

    public String[] getSentencesOfDocumentAsStringArray(Document document){
        return document.streamSentences()
                .map(Sentence::getText)
                .toArray(String[]::new);
    }

    public String[] getTokensOfDocumentAsStringArray1D(Document document){
        return document.streamTokens()
                .map(Token::getText)
                .toArray(String[]::new);
    }

    public String[][] getTokensOfDocumentAsStringArray2D(Document document){
        return document.streamSentences()
                .map(this::getTokensOfSentenceAsStringArray)
                .toArray(String[][]::new);
    }

    public void putVectorInToken(Token token, double[] data){
        token.putVector(getClass(), Nd4j.create(data, new long[]{getEmbeddingVectorSize(), 1}));
    }

    public void putVectorInSentence(Sentence sentence, double[] data){
        sentence.putVector(getClass(), Nd4j.create(data, new long[]{getEmbeddingVectorSize(), 1}));
    }

    public void putVectorInTokenOfSentence(Sentence sentence, double[][] data){
        int i=0;
        for(Token token: sentence.getTokens()){
            token.putVector(getClass(), Nd4j.create(data[i++], new long[]{getEmbeddingVectorSize(), 1}));
        }
    }

    public void putVectorInDocument(Document document, double[] data){
        document.putVector(getClass(), Nd4j.create(data, new long[]{getEmbeddingVectorSize(), 1}));
    }

    public void putVectorInSentenceOfDocument(Document document, double[][] data){
        int i=0;
        for(Sentence sentence: document.getSentences()){
            sentence.putVector(getClass(), Nd4j.create(data[i++], new long[]{getEmbeddingVectorSize(), 1}));
        }
    }

    public void putVectorInTokenOfDocument1D(Document document, double[][] data){
        int i=0;
        for(Token token: document.getTokens()){
            token.putVector(getClass(), Nd4j.create(data[i++], new long[]{getEmbeddingVectorSize(), 1}));
        }
    }

    public void putVectorInTokenOfDocument2D(Document document, double[][][] data){
        int i=0;
        for(Sentence sentence: document.getSentences()){
            int n=0;
            for(Token token: sentence.getTokens()){
                token.putVector(getClass(), Nd4j.create(data[i][n++], new long[]{getEmbeddingVectorSize(), 1}));
            }
            i++;
        }
    }
}
