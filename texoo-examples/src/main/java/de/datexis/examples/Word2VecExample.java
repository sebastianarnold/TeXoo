package de.datexis.examples;

import de.datexis.common.DialogHelpers;
import de.datexis.encoder.impl.Word2VecEncoder;
import java.util.Collection;


import de.datexis.common.Resource;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Example: how to use a pretrained Word2Vec model
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class Word2VecExample {

  protected final static Logger log = LoggerFactory.getLogger(Word2VecExample.class);

  /**
   * This example requires a word2vec model stored on disk.
   * @param args : [0] full path to word2vec model, e.g. /home/texoo/Word2VecEncoder_en_150_Wiki+minlc.bin
   */
  public static void main(String[] args) throws IOException {
    String modelDir = args.length > 0 ? args[0] : 
        DialogHelpers.askForDirectory("Please choose a word2vec model directory");
    new Word2VecExample().tryWord2Vec(modelDir);
  }
    
   public void tryWord2Vec(String modelDir) throws IOException {
     
    Resource vectorModel = Resource.fromFile(modelDir);
    Word2VecEncoder vec = Word2VecEncoder.load(vectorModel);    
    
    // Find similar words with a single-word query:
    //List<String> words = Arrays.asList("David", "Belin", "berlin", "Obama", "fährt", "und", "ich", "Hochschule", "Microsoft", "Google");
    List<String> words = Arrays.asList("David", "berlin", "Obama", "drives", "buy", "you", "University", "Microsoft", "Google");
    for(String w : words) {
      Collection<String> lst = vec.getNearestNeighbours(w, 50);
      System.out.println("Closest Words to \"" + w + "\": " + lst.toString());
    }
    
    // To look for similar words with a multi-word query, we can encode a vector as search key:
    List<String> multiwords = Arrays.asList("Microsoft Word", "Barack Obama");
    for(String w : multiwords) {
      INDArray key = vec.encode(w); // will create a mixture model vector
      Collection<String> lst = vec.getNearestNeighbours(key, 50);
      System.out.println("Closest Words to \"" + w + "\": " + lst.toString());
    } 
    
  }
  
}
