package de.datexis.cdv.index;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import de.datexis.cdv.model.EntityAspectAnnotation;
import de.datexis.cdv.preprocess.AspectPreprocessor;
import de.datexis.common.Resource;
import de.datexis.common.WordHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.model.*;
import de.datexis.preprocess.DocumentFactory;
import de.datexis.retrieval.tagger.LSTMSentenceTaggerIterator;
import de.datexis.retrieval.tagger.LabeledSentenceIterator;
import de.datexis.tagger.AbstractMultiDataSetIterator;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * An index that holds aspects (e.g. "signs_and_symptoms") as keys and vectors as values.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class AspectIndex extends QueryIndex {
  
  protected final static Logger log = LoggerFactory.getLogger(AspectIndex.class);
  
  public static String HEADING_SEPARATOR_REGEX = " \\| | and |&|/";
  
  /** called from JSON deserialization */
  protected AspectIndex() {}
  
  public AspectIndex(IEncoder encoder) {
    super(new AspectPreprocessor(), encoder);
    this.id = "ASP";
  }
  
  @Override
  public INDArray encode(Span span) {
    if(span instanceof Token) return encode(span.getText());
    else throw new IllegalArgumentException("Index is not configured to encode " + span.getClass());
  }
  
  /**
   * Encode a heading into a dense vector. Will split headings at " | " characters and calculate average vector.
   */
  @Override
  public INDArray encode(String heading) {
    String[] headings = heading.split(HEADING_SEPARATOR_REGEX);
    INDArray avg = Nd4j.zeros(DataType.FLOAT, getEmbeddingVectorSize(), 1);
    INDArray vec;
    int count = 0;
    for(String s : headings) {
      vec = super.encode(s.trim());
      if(vec != null) {
        avg.addi(vec);
        count++;
      }
    }
    return count > 1 ? avg.divi(count) : avg;
  }
  
  /**
   * Lookup a heading into a dense vector. Will split headings at " | " characters and calculate average vector.
   */
  @Override
  public INDArray lookup(String heading) {
    String[] headings = heading.split(HEADING_SEPARATOR_REGEX);
    INDArray vec, sum = Nd4j.zeros(DataType.FLOAT, getEmbeddingVectorSize(), 1);
    int count = 0;
    for(String s : headings) {
      vec = super.lookup(s.trim());
      if(vec != null) {
        sum.addi(vec);
        count++;
      }
    }
    if(count == 0) return null;
    else return count > 1 ? sum.divi(count) : sum;
  }
  
  /**
   * Encode a heading into a sparse vector. Will split headings at " | " characters and calculate average vector.
   */
  public INDArray decode(String heading) {
    String[] headings = heading.split(HEADING_SEPARATOR_REGEX);
    INDArray result = Nd4j.zeros(DataType.FLOAT, size(), 1);
    for(String key : headings) {
      int idx = index(key.trim());
      if(idx >= 0) result.putScalarUnsafe(idx, 1.);
      //else log.warn("split '{}' not contained in index", key);
    }
    if(result.maxNumber().intValue() == 0) {
      int idx = findIndex(encode(heading));
      if(idx >= 0) result.putScalarUnsafe(idx, 1.);
      log.warn("heading '{}' not contained in index, using nearest neighbour '{}'", heading, key(idx));
    }
    return result;
  }
  
  /**
   * Train a model by averaging over all labels in a given TSV file <label>\tab<sentence>
   */
  public void encodeIndexFromLabels(Resource sentencesTSV) {
    LabeledSentenceIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.ENCODE, encoder, null, sentencesTSV, "utf-8", WordHelpers.Language.EN, true, 64);
    List<String> labels = it.getLabels();
    Multimap<String, Sentence> examples = ArrayListMultimap.create();
    List<String> keys = new ArrayList<>();
    for(String label : labels) {
      if(label.equals("Abstract")) label = "Description";
      String[] splits = label.split(HEADING_SEPARATOR_REGEX);
      for(String s : splits) {
        String key = keyPreprocessor.preProcess(s.trim());
        Sentence span = DocumentFactory.createSentenceFromTokenizedString(s.trim());
        keys.add(key);
        if(!examples.containsKey(key)) // we need to encode every label only once
          examples.put(key, span); // just a simple mapping from heading to heading!
      }
    }
    buildKeyIndex(keys, false);
    encodeAndBuildVectorIndex(examples, false);
    setModelAvailable(true);
  }
  
  /**
   * Train a model by averaging over all labels in a given TSV file <label>\tab<sentence>
   */
  public void encodeIndexFromSentences(Resource sentencesTSV, Set<String> stopWords, boolean isTokenized) {
    LabeledSentenceIterator it = new LSTMSentenceTaggerIterator(AbstractMultiDataSetIterator.Stage.ENCODE, encoder, null, sentencesTSV, "utf-8", WordHelpers.Language.EN, stopWords, isTokenized, 1);
    log.info("Reading {} examples...", it.getNumExamples());
    Multimap<String, Sentence> examples = ArrayListMultimap.create();
    String key;
    while(it.hasNext()) {
      Map.Entry<String, Sentence> example = it.nextLabeledSentence();
      key = example.getKey();
      if(key.equals("Abstract")) key = "Description";
      String[] splits = key.split(HEADING_SEPARATOR_REGEX);
      for(String s : splits) {
        String k = keyPreprocessor.preProcess(s.trim());
        examples.put(k, example.getValue());
      }
    }
    buildKeyIndex(examples.keys(), false);
    encodeAndBuildVectorIndex(examples, false);
    setModelAvailable(true);
  }
  
  @Override
  public void encodeFromQueries(Collection<Query> queries) {
  
  }
  
  /**
   * Build an index over headings given by GOLD EntityAspectAnnotations in the Documents.
   */
  @Override
  @Deprecated
  public void trainModel(Collection<Document> documents) {
    List<String> keys = new ArrayList<>();
    Map<String, INDArray> vectors = new HashMap<>();
    for(Document doc : documents) {
      for(EntityAspectAnnotation ann : doc.getAnnotations(Annotation.Source.GOLD, EntityAspectAnnotation.class)) {
        String aspect = ann.getAspect();
        if(aspect != null) {
          String[] splits = ann.getLabel().split(HEADING_SEPARATOR_REGEX);
          for(String split : splits) {
            String key = keyPreprocessor.preProcess(split);
            keys.add(key);
            if(!vectors.containsKey(key)) {
              Sentence asp = DocumentFactory.createSentenceFromTokenizedString(split);
              INDArray vec = encoder.encode(asp);
              vectors.put(key, vec);
            }
          }
        }
      }
    }
    buildKeyIndex(keys, false);
    buildVectorIndex(vectors, false);
    setModelAvailable(true);
  }
  
}
