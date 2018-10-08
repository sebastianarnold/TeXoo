package de.datexis.encoder.impl;

import de.datexis.encoder.StaticEncoder;
import de.datexis.model.Span;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
public class CharClassEncoder extends StaticEncoder {
  
  public CharClassEncoder() {
    super("CHR");
    log = LoggerFactory.getLogger(CharClassEncoder.class);
  }
  
  public CharClassEncoder(String id) {
    super(id);
    log = LoggerFactory.getLogger(CharClassEncoder.class);
  }
  
  @Override
  public String getName() {
    return "Character Class AbstractEncoder";
  }

  @Override
  public long getVectorSize() {
    return encode("Test").length();
  }

  @Override
  public INDArray encode(Span span) {
    return encode(span.getText());
  }
  
  @Override
  public INDArray encode(String span) {
   
    span = span.trim();
    ArrayList<Boolean> features = new ArrayList<>();
    // Character classes, also see http://www.regular-expressions.info/unicode.html#category
    features.add(isSymbol(span));
    features.add(isOperator(span));
    features.add(isOpeningQuote(span));
    features.add(isClosingQuote(span));
    features.add(isOpeningParanthesis(span));
    features.add(isClosingParanthesis(span));
    features.add(isSlash(span));
    features.add(isComma(span));
    features.add(isDash(span));
    features.add(isSentenceTerminator(span));
    features.add(isColon(span));
    
    INDArray vector = Nd4j.zeros(features.size(), 1);
    int i = 0;
    for(Boolean f : features) {
      vector.put(i++, 0, f ? 1.0 : 0.0);
    }
    return vector;
  }
  
  // symbols and signs    SYM $ #
  static Collection<String> symbols = Arrays.asList("#","$","%","@","^","_","~","¢","£","¥","§","€");
  public boolean isSymbol(String token) {
    return symbols.contains(token);
  }
  // operators  + - & *
  static Collection<String> operators = Arrays.asList("&","*","+","=");
  public boolean isOperator(String token) {
    return operators.contains(token);
  }
  // opening quotes           ` ``
  // other quotes             "
  static Collection<String> oquotes = Arrays.asList("\"","`","``");
  public boolean isOpeningQuote(String token) {
    return symbols.contains(token);
  }
  // closing quotes           ' ''
  static Collection<String> cquotes = Arrays.asList("'","''");
  public boolean isClosingQuote(String token) {
    return symbols.contains(token);
  }
  // opening paranthesis      ( [ {
  // -LRB- -END-?
  static Collection<String> oparanthesis = Arrays.asList("(","<","[","{","-LRB-");
  public boolean isOpeningParanthesis(String token) {
    return oparanthesis.contains(token);
  }
  // closing paranthesis      ) ] }
  static Collection<String> cparanthesis = Arrays.asList(")",">","]","}","-RRB-");
  public boolean isClosingParanthesis(String token) {
    return cparanthesis.contains(token);
  }
  // slashes                  / \ |
  static Collection<String> slashes = Arrays.asList("/","\\","|");
  public boolean isSlash(String token) {
    return slashes.contains(token);
  }
  // comma                    ,
  static Collection<String> commas = Arrays.asList(",");
  public boolean isComma(String token) {
    return commas.contains(token);
  }
  // dashes                   -- - – 
  // list item marker     LS
  static Collection<String> dashes = Arrays.asList("-","–","--","---");
  public boolean isDash(String token) {
    return dashes.contains(token);
  }
  // sentence terminator      . ! ? 
  static Collection<String> sterminator = Arrays.asList(".","!","?");
  public boolean isSentenceTerminator(String token) {
    return sterminator.contains(token);
  }
  // colons and ellipses      : ; ... 
  static Collection<String> colons = Arrays.asList(":",";","...");
  public boolean isColon(String token) {
    return colons.contains(token);
  }

  // don't encode any closed word classes (language dependent!):
  // conjunction          IN
  // determiner           DT
  // preposition          IN
  // possessive ending    POS
  // wh-pronoun           W*
  
}
