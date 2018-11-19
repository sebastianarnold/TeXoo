package de.datexis.encoder.impl;

import cc.fasttext.FastText;
import cc.fasttext.Vector;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import de.datexis.common.*;
import de.datexis.encoder.Encoder;
import de.datexis.preprocess.MinimalLowercasePreprocessor;
import de.datexis.model.*;
import de.datexis.model.Token;
import java.io.IOException;
import java.util.Collection;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Interface to FastText port from https://github.com/sszuev/fastText_java
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class FastTextEncoder extends Encoder {

	private static final Logger log = LoggerFactory.getLogger(FastTextEncoder.class);

	private FastText ft;
  private String modelName;
  private static final TokenPreProcess preprocessor = new MinimalLowercasePreprocessor();
  
	public FastTextEncoder() {
    super("FT");
  }
  
  public FastTextEncoder(String id) {
    super(id);
  }

  public static FastTextEncoder load(Resource path) throws IOException {
    FastTextEncoder ft = new FastTextEncoder();
    ft.loadModel(path);
    return ft;
  }
  
  @Override
  public void loadModel(Resource modelFile) throws IOException {
    log.info("Loading FastText model: " +  modelFile.getFileName());
    ft = FastText.load(modelFile.toString());
    //setModel(modelFile);
    setModelAvailable(true);
    log.info("Loaded FastText model {} with nlabels={}, ntokens={}, nwords={}, size={}, '", 
      modelFile.getFileName(),
      ft.getDictionary().nlabels(),
      ft.getDictionary().ntokens(),
      ft.getDictionary().nwords(),
      ft.getDictionary().size()
    );
	}
  
  @Override
  public void saveModel(Resource modelPath, String name) {
    throw new UnsupportedOperationException("model saving not implemented");
  }
  
  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("model training not implemented");
  }
  
	@Override
	public String getName() {
		return modelName;
	}
  
  protected INDArray asINDArray(Vector vec) {
    INDArray arr = Nd4j.createUninitialized(vec.size(), 1);
    int i = 0;
    for(Float f : vec.getData()) {
      arr.putScalar(i++, 1, f.doubleValue());
    }
    return arr;
  }
  
	/**
	 * Use this function to access word vectors
	 * @param word
	 * @return
	 */
	private INDArray getWordVector(String word) {
		return asINDArray(ft.getWordVector(word));//(preprocessor.preProcess(word));
	}

	public boolean isUnknown(String word) {
		return ft.getDictionary().getId(word) <= 0;
	}

	@Override
	public INDArray encode(Span span) {
    if(span instanceof Token) return encode(preprocessor.preProcess(span.getText()));
    else return encode(span.getText());
	}

	@Override
	public long getEmbeddingVectorSize() {
		return ft.getWordVector("the").size();
	}

  /**
   * Encodes the word. Returns nullvector if word was not found.
   * @param word
   * @return 
   */
	@Override
	public INDArray encode(String word) {
    return getWordVector(word);
	}

	public Collection<String> getNearestNeighbours(String word, int k) {
		return ft.nn(k, word).keySet();
	}

	public Collection<String> getNearestNeighbours(INDArray v, int k) {
		throw new UnsupportedOperationException("not implemented");
	}

	public String getNearestNeighbour(INDArray v) {
		Collection<String> result = getNearestNeighbours(v, 1);
    if(result.isEmpty()) return "_";
    else return result.iterator().next();
  }

}
