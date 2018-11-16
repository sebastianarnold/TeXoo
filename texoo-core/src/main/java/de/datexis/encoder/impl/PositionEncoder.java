package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Token;
import de.datexis.encoder.StaticEncoder;
import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Iterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.slf4j.LoggerFactory;

/**
 * Encodes several features of the word's surface form. Please note that we do not encode any
 * language specific features, such as closed word classes. These classes should be detected
 * by SkipGramEncoder depending on the language.
 * Some Penn Treebank Part-of-Speech Tags overviews and extensions can be found here:
 * https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
 * http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/Penn-Treebank-Tagset.pdf
 * https://www.eecis.udel.edu/~vijay/cis889/ie/pos-set.pdf
 * https://www.comp.leeds.ac.uk/ccalas/tagsets/upenn.html
 * http://erwinkomen.ruhosting.nl/eng/2014_Longdale-Labels.htm
 * @author sarnold
 */
public class PositionEncoder extends StaticEncoder {
  
  public PositionEncoder() {
    super("POS");
    log = LoggerFactory.getLogger(PositionEncoder.class);
  }
  
  public PositionEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(PositionEncoder.class);
  }
  
  @Override
  public String getName() {
    return "Positional Encoder";
  }

  @Override
  @JsonIgnore
  public long getEmbeddingVectorSize() {
    return wordAsVector("", false, false, false, false).length();
  }
  
  public void setVectorSize(int size) {
    if(size != getEmbeddingVectorSize()) {
      throw new IllegalArgumentException("Vector size of saved Encoder (" + getEmbeddingVectorSize() + ") differs from implementation (" + size + ")");
    }
  }

  @Override
  public INDArray encode(Span span) {
    throw new IllegalArgumentException("PositionEncoder is sequential, you need to call encodeEach()");
  }
  
  @Override
  public INDArray encode(String token) {
    throw new IllegalArgumentException("PositionEncoder is sequential, you need to call encodeEach()");
  }
  
  public INDArray tokenAsVector(Token token, boolean beginDoc, boolean beginSent, boolean endSent, boolean endDoc) {
    return wordAsVector(token.getText(), beginDoc, beginSent, endSent, endDoc);
  }
  
  public INDArray wordAsVector(String token, boolean beginDoc, boolean beginSent, boolean endSent, boolean endDoc) {
    ArrayList<Boolean> features = new ArrayList<>();
    features.add(beginDoc); // begin of document
    features.add(beginSent); // begin of sentence
    features.add(endSent); // end of sentence
    features.add(endDoc); // end of document
    INDArray vector = Nd4j.zeros(features.size(), 1);
    int i = 0;
    for(Boolean f : features) {
      vector.put(i++, 0, f ? 1.0 : 0.0);
    }
    return vector;
  }
  
  @Override
  public void encodeEach(Document d, Class<? extends Span> elementClass) {
    if(elementClass != Token.class) throw new IllegalArgumentException("PositionEncoder is only implemented to encode Tokens over Documents.");
    boolean beginDoc = true, beginSent, endSent, endDoc;
    Iterator<Sentence> sentences = d.getSentences().iterator();
    while(sentences.hasNext()) {
      Sentence s = sentences.next();
      endDoc = !sentences.hasNext();
      beginSent = true;
      Iterator<Token> tokens = s.getTokens().iterator();
      while(tokens.hasNext()) {
        Token t = tokens.next();
        endSent = !tokens.hasNext();
        t.putVector(PositionEncoder.class, tokenAsVector(t, beginDoc && beginSent, beginSent, endSent, endDoc && endSent));
        beginSent = false;
      }
      beginDoc = false;
    }
  }

  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    throw new IllegalArgumentException("PositionEncoder is only implemented to encode Tokens over Documents.");
  }
  
}
