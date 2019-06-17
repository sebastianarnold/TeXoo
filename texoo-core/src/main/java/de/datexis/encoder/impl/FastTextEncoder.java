package de.datexis.encoder.impl;

import cc.fasttext.FastText;
import cc.fasttext.Matrix;
import cc.fasttext.Vector;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Multimaps;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.model.Token;
import de.datexis.preprocess.SentenceDetectorMENL;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.Validate;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;
import java.util.logging.Level;
import java.util.stream.Collectors;

/**
 * Interface to FastText port from https://github.com/sszuev/fastText_java
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class FastTextEncoder extends Encoder {

	private static final Logger log = LoggerFactory.getLogger(FastTextEncoder.class);

	private FastText ft;
  private String modelName;
  private Resource modelSource;
  private long size = 0;

  private Method getPrecomputedWordVectors, findNN;

	public FastTextEncoder() {
    super("FT");
  }
  
  public FastTextEncoder(String id) {
    super(id);
  }

  private void initializeMethodsFromReflection() {
    try {
      // we need to get some methods from FastText through reflection
      getPrecomputedWordVectors = ft.getClass().getDeclaredMethod("getPrecomputedWordVectors");
      getPrecomputedWordVectors.setAccessible(true);
      findNN = ft.getClass().getDeclaredMethod("findNN", Matrix.class, Vector.class, int.class, Set.class);
      findNN.setAccessible(true);
      /*for(Method method : ft.getClass().getDeclaredMethods()) {
        if(method.getName().equals("getPrecomputedWordVectors")) {
          getPrecomputedWordVectors = method;
        } else if(method.getName().equals("findNN")) {
          findNN = method;
        }
      }*/
    } catch(Exception ex) {
      java.util.logging.Logger.getLogger(SentenceDetectorMENL.class.getName()).log(Level.SEVERE, null, ex);
    }
  }

  public static FastTextEncoder load(Resource path) throws IOException {
    FastTextEncoder ft = new FastTextEncoder();
    ft.loadModel(path);
    return ft;
  }
  
  /**
   * Load an existing FastText binary model and keep a copy. On save, the model will be duplicated to Annotator folder.
   */
  @Override
  public void loadModel(Resource modelFile) throws IOException {
    log.info("Loading FastText model: " +  modelFile.getFileName());
    ft = FastText.DEFAULT_FACTORY.load(modelFile.getInputStream());
    initializeMethodsFromReflection();
    size = ft.getWordVector("the").size();
    setModel(modelFile);
    setModelAvailable(true);
    modelSource = modelFile;
    log.info("Loaded FastText model '{}' with {} words and vector size {}",
              modelFile.getFileName(), ft.getDictionary().size(), size);
	}
  
  /**
   * Load an existing FastText binary model and keep its reference. On save, the model will not be copied.
   */
	public void loadModelAsReference(Resource modelFile) throws IOException {
	  loadModel(modelFile);
    modelSource = null;
  }
  
  @Override
  public void saveModel(Resource modelPath, String name) {
     if(modelSource != null) {
       try {
         Resource modelFile = modelPath.resolve(name + (modelSource.getFileName().endsWith(".gz") ? ".bin.gz" : ".bin"));
         FileUtils.copyFile(modelSource.toFile(), modelFile.toFile());
         setModel(modelPath);
       } catch(IOException ex) {
         log.error(ex.toString());
       }
     } // else rely on AnnotatorFactory to find the model in the search path
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("model training not implemented");
  }
  
	@Override
	public String getName() {
		return modelName;
	}
  
  /**
   * @return FastText Vector as INDArray
   */
  protected INDArray asINDArray(Vector vec) {
    INDArray arr = Nd4j.createUninitialized(vec.size(), 1);
    int i = 0;
    for(Float f : vec.getData()) {
      arr.putScalar(i++, 1, f.doubleValue());
    }
    return arr;
  }

  /**
   * @return INDArray as FastText Vector
   */
  protected Vector asVector(INDArray arr) {
    Vector vec = new Vector((int)arr.length());
    for(int i = 0; i < arr.length(); i++) {
      vec.set(i, arr.getFloat(i));
    }
    return vec;
  }
  
	/**
	 * Use this function to access word vectors
	 */
	protected INDArray getWordVector(String word) {
		return asINDArray(ft.getWordVector(word));
	}
  
  /**
	 * Use this function to access sentence vectors
	 */
	protected INDArray getSentenceVector(String sentence) {
		return asINDArray(ft.getSentenceVector(sentence));
	}

	public boolean isUnknown(String word) {
		return ft.getDictionary().getId(word) <= 0;
	}

	@Override
	public INDArray encode(Span span) {
    if(span instanceof Token) return getWordVector(span.getText());
    else if(span instanceof Sentence) return getSentenceVector(((Sentence) span).toTokenizedString());
    else return encode(span.getText());
	}

	@Override
	public long getEmbeddingVectorSize() {
		return size;
	}

  /**
   * Encodes the word. Returns nullvector if word was not found.
   */
	@Override
	public INDArray encode(String word) {
	  if(word.contains(" ")) return getSentenceVector(word);
	  else return getWordVector(word);
	}

	public List<String> getNearestNeighbours(String word, int k) {
    Multimap<String, Float> result = ft.nn(k, word);
    return result.entries().stream()
      .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
      .map(e -> e.getKey())
      .collect(Collectors.toList());
	}

	public List<String> getNearestNeighbours(INDArray v, int k) {
    try {
      Validate.isTrue(k > 0, "Not positive factor");
      Matrix wordVectors = null;
      wordVectors = (Matrix) getPrecomputedWordVectors.invoke(ft);
      Set<String> banSet = new HashSet<>();
      Vector queryVec = asVector(v);
      Multimap<String, Float> result = Multimaps.invertFrom((Multimap<Float, String>) findNN.invoke(ft, wordVectors, queryVec, k, banSet), ArrayListMultimap.create());
      return result.entries().stream()
        .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
        .map(e -> e.getKey())
        .collect(Collectors.toList());
    } catch(IllegalAccessException | InvocationTargetException e) {
      e.printStackTrace();
    }
    return Collections.EMPTY_LIST;
  }

}
