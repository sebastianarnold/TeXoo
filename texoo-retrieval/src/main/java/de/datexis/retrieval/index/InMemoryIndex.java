package de.datexis.retrieval.index;

import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import de.datexis.common.Resource;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncodingHelpers;
import de.datexis.encoder.IEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import de.datexis.preprocess.IdentityPreprocessor;
import de.datexis.retrieval.encoder.LSTMSentenceEncoder;
import de.datexis.retrieval.tagger.LSTMSentenceTagger;
import org.apache.commons.lang.Validate;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * An in-memory index that stores key-value pairs and allows KNN queries
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class InMemoryIndex extends Encoder implements IEncoder, IVocabulary, IVectorIndex {

  protected final static Logger log = LoggerFactory.getLogger(InMemoryIndex.class);

  /** Encoder to generate embeddings */
  protected IEncoder encoder;

  /** Underlying vocabulary cache */
  protected AbstractCache<VocabWord> keyVocabulary;

  /** Lookup table for id -> vector */
  protected InMemoryLookupTable<VocabWord> lookupVectors;
  
  /** Preprocessor for lookup keys */
  protected TokenPreProcess keyPreprocessor;
  
  /** called from JSON deserialization */
  protected InMemoryIndex() {}
  
  public InMemoryIndex(IEncoder encoder) {
    this(new IdentityPreprocessor(), encoder);
  }
  
  public InMemoryIndex(TokenPreProcess keyPreprocessor, IEncoder encoder) {
    super("KNN");
    this.encoder = encoder;
    this.keyPreprocessor = keyPreprocessor;
    this.keyVocabulary = new AbstractCache.Builder()
            .hugeModelExpected(false)
            .minElementFrequency(0)
            .build();
    this.lookupVectors = new InMemoryLookupTable<>(keyVocabulary, (int)getEmbeddingVectorSize(), true, 0.01, Nd4j.getRandom(), 0, true);
  }

  // --- model training (extends Encoder) ------------------------------------------------------------------------------

  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("not implemented");
  }
  
  public void buildKeyIndex(Iterable<String> keys) {
    buildKeyIndex(keys, true);
  }
  
  public void buildKeyIndex(Iterable<String> keys, boolean normalizeKeys) {
    log.info("Building key index...");
    int idx = 0;
    // fill keyVocabulary
    for(String key : keys) {
      if(normalizeKeys) key = keyPreprocessor.preProcess(key);
      if(!keyVocabulary.containsWord(key)) {
        VocabWord word = new VocabWord(1.0, key, idx++);
        word.setSpecial(false);
        word.markAsLabel(true);
        word.setIndex(keyVocabulary.numWords());
        keyVocabulary.addToken(word);
        keyVocabulary.addWordToIndex(word.getIndex(), word.getLabel());
      } else {
        keyVocabulary.incrementWordCount(key);
      }
    }
  }

  public void buildVectorIndex(Map<String, INDArray> vectors) {
    log.info("Building vector index for {} entries...", vectors.size());
    if(size() <= 0) throw new IllegalStateException("Cannot insert vectors into empty index. Please insert keys first.");
    lookupVectors.resetWeights();
    long num = 0;
    for(Map.Entry<String, INDArray> vec : vectors.entrySet()) {
      lookupVectors.putVector(keyPreprocessor.preProcess(vec.getKey()), vec.getValue());
      if(++num % 100000 == 0) log.info("inserted {} vectors into vector index", num);
    }
    // apply normalization
    INDArray syn0 = lookupVectors.getSyn0();
    syn0.diviColumnVector(syn0.norm2(1));
  }
  
  /**
   * Use given entries to build a vector index. All descriptions are encoded into vectors
   * using the configured Encoder.
   * @param entries Map key -> description
   */
  public void encodeAndBuildVectorIndex(Map<String, String> entries) {
    log.info("Building vector index for {} entries...", entries.size());
    if(size() <= 0) throw new IllegalStateException("Cannot insert vectors into empty index. Please insert keys first.");
    lookupVectors.resetWeights();
    long num = 0;
    for(Map.Entry<String, String> span : entries.entrySet()) {
      INDArray vec = encoder.encode(span.getValue()); // not normalized yet
      lookupVectors.putVector(keyPreprocessor.preProcess(span.getKey()), vec);
      if(++num % 100000 == 0) log.info("inserted {} vectors into vector index", num);
    }
    // apply normalization
    INDArray syn0 = lookupVectors.getSyn0();
    syn0.diviColumnVector(syn0.norm2(1));
  }
  
  /**
   * Use given entries to build a vector index. All descriptions are encoded into average vectors
   * using the configured Encoder.
   * @param examples Map key -> multiple descriptions (key needs to be normalized already!)
   */
  public void encodeAndBuildVectorIndex(Multimap<String, ? extends Span> examples, boolean lookup) {
    log.info("Building vector index for {} entries...", examples.keySet().size());
    if(size() <= 0) throw new IllegalStateException("Cannot insert vectors into empty index. Please insert keys first.");
    lookupVectors.resetWeights();
    long num = 0;
    Nd4j.getMemoryManager().togglePeriodicGc(false);
    // first encode the examples in batches
    Map<String, INDArray> sums = new HashMap<>(this.size());
    List<List<Map.Entry<String, ? extends Span>>> batches = Lists.partition(Lists.newArrayList(examples.entries()), 128);
    for(List<Map.Entry<String, ? extends Span>> batch : batches) {
      INDArray embs, vec;
      if(encoder instanceof LSTMSentenceEncoder) {
        LSTMSentenceTagger tagger = ((LSTMSentenceEncoder) encoder).getTagger();
        List<Sentence> sents = batch.stream().map(b -> (Sentence) b.getValue()).collect(Collectors.toList());
        embs = tagger.encodeBatchMatrix(sents);
      } else {
        List<? extends Span> sents = batch.stream().map(b -> b.getValue()).collect(Collectors.toList());
        embs = lookup && encoder instanceof InMemoryIndex ?
          lookupBatchMatrix(sents, (InMemoryIndex) encoder) :
          EncodingHelpers.encodeBatchMatrix(sents, encoder);
      }
      for(int batchNum = 0; batchNum < batch.size(); batchNum++) {
        String key = batch.get(batchNum).getKey();
        INDArray sum = sums.getOrDefault(key, Nd4j.zeros(DataType.FLOAT, encoder.getEmbeddingVectorSize(), 1));
        vec = embs.getRow(batchNum).reshape(encoder.getEmbeddingVectorSize(), 1);
        sum.addi(vec);
        sums.put(key, sum);
        if(++num % 10000 == 0) log.info("encoded {} vectors", num);
      }
    }
    // then put averages into index
    num = 0;
    for(String key : examples.keySet()) {
      INDArray sum = sums.get(key);
      int count = examples.get(key).size();
      if(sum != null) {
        lookupVectors.putVector(key, count > 1 ? sum.divi(count) : sum);
        if(++num % 100000 == 0) log.info("inserted {} vectors into vector index", num);
      }
    }
    // apply normalization
    INDArray syn0 = lookupVectors.getSyn0();
    syn0.diviColumnVector(syn0.norm2(1));
  }
  
  public static INDArray lookupBatchMatrix(List<? extends Span> input, InMemoryIndex index) {
    INDArray encoding = Nd4j.zeros(DataType.FLOAT, input.size(), index.getEmbeddingVectorSize());
    Span example;
    for(int batchIndex = 0; batchIndex < input.size(); batchIndex++) {
      example = input.get(batchIndex);
      INDArray vec = index.lookup(example.getText());
      encoding.get(point(batchIndex), all()).assign(vec);
    }
    return encoding;
  }

  @Override
  public void saveModel(Resource path, String name) throws IOException {
    Resource modelFile = path.resolve(name + ".bin");
    writeBinaryModel(modelFile.getOutputStream());
    setModel(modelFile);
  }
  
  @Override
  public void loadModel(Resource modelFile) throws IOException {
    loadBinaryModel(modelFile.getInputStream());
    setModel(modelFile);
    setModelAvailable(true);
  }
  
  /**
   * Writes the model to DATEXIS binary format
   */
  private void writeBinaryModel(OutputStream outputStream) throws IOException {
  
    int keys = 0;
    
    try(BufferedOutputStream buf = new BufferedOutputStream(outputStream);
        DataOutputStream writer = new DataOutputStream(buf)) {
      
      // write header
      int numWords = keyVocabulary.numWords();
      writer.writeLong(numWords);
      writer.writeLong(keyVocabulary.totalNumberOfDocs());
      writer.writeLong(lookupVectors.layerSize());
      
      // write words
      for(int i = 0; i < numWords; i++) {
        VocabWord word = keyVocabulary.elementAtIndex(i);
        writer.writeUTF(word.getLabel());
        writer.writeDouble(word.getElementFrequency());
        keys++;
      }
      
      // write vectors
      for(int i = 0; i < numWords; i++) {
        VocabWord word = keyVocabulary.elementAtIndex(i);
        INDArray vec = lookupVectors.vector(word.getLabel());
        Nd4j.write(vec, writer);
      }
      
      writer.flush();
      
    }
    
    log.info("Wrote {} entries with vector size {}", keys, lookupVectors.layerSize());
    
  }
  
  /**
   * Loads the model from DATEXIS bindary format
   */
  private void loadBinaryModel(InputStream stream) throws IOException {
  
    int idx = 0;
  
    try(BufferedInputStream buf = new BufferedInputStream(stream);
        DataInputStream reader = new DataInputStream(buf)) {
      
      long numWords = reader.readLong();
      long numDocs = reader.readLong();
      long layerSize = reader.readLong();
      
      this.keyVocabulary = new AbstractCache.Builder()
            .hugeModelExpected(false)
            .minElementFrequency(0)
            .build();
      this.lookupVectors = new InMemoryLookupTable<>(keyVocabulary, (int)layerSize, true, 0.01, Nd4j.getRandom(), 0, true);
      
      // load words
      for(int i = 0; i < numWords; i++) {
        if(reader.available() <= 0) throw new IOException("binary file truncated");
        String key = reader.readUTF();
        double freq = reader.readDouble();
        VocabWord word = new VocabWord(freq, key, idx++);
        word.setSpecial(false);
        word.markAsLabel(true);
        word.setIndex(keyVocabulary.numWords());
        keyVocabulary.addToken(word);
        keyVocabulary.addWordToIndex(word.getIndex(), word.getLabel());
        if(idx % 100000 == 0) log.info("loaded {} keys into word index", i);
      }
      keyVocabulary.updateWordsOccurrences();
  
      // load vectors
      idx = 0;
      this.lookupVectors.resetWeights();
      for(int i = 0; i < numWords; i++) {
        if(reader.available() <= 0) throw new IOException("binary file truncated");
        INDArray vec = Nd4j.read(reader);
        lookupVectors.putVector(keyVocabulary.wordAtIndex(i), vec);
        if(++idx % 100000 == 0) log.info("loaded {} vectors into vector index", idx);
      }
      
      // INDArray syn0 = lookupVectors.getSyn0();
      // syn0.diviColumnVector(syn0.norm2(1));
    }
  
    log.info("Read {} entries with vector size {}", idx, lookupVectors.layerSize());
    
  }
  
  @JsonIgnore
  public IEncoder getEncoder() {
    return encoder;
  }
  
  public void setEncoder(IEncoder encoder) {
    this.encoder = encoder;
  }
  
  @JsonTypeInfo(use=JsonTypeInfo.Id.CLASS, include=JsonTypeInfo.As.PROPERTY, property="class")
  public TokenPreProcess getKeyPreprocessor() {
    return keyPreprocessor;
  }
  
  public void setKeyPreprocessor(TokenPreProcess keyPreprocessor) {
    this.keyPreprocessor = keyPreprocessor;
  }
  
  @Override
  public void setEncoders(List<Encoder> encoders) {
    if(encoders.size() != 1)
      throw new IllegalArgumentException("wrong number of encoders given (expected=1, actual=" + encoders.size() + ")");
    encoder = encoders.get(0);
  }
  
  @Override
  public List<Encoder> getEncoders() {
    return Lists.newArrayList((Encoder)encoder);
  }
  
  // --- text encoding methods to produce embeddings (implements IEncoder) ---------------------------------------------

  /**
   * @return size of the dense vector representation K
   */
  @Override
  public long getEmbeddingVectorSize() {
    return encoder.getEmbeddingVectorSize();
  }

  /**
   * @return {0...1}^K dense vector representation of the given phrase
   */
  @Override
  public INDArray encode(String word) {
    return Transforms.unitVec(encoder.encode(word));
  }

  /**
   * @return {0...1}^K dense vector representation of the given phrase
   */
  @Override
  public INDArray encode(Span span) {
    return Transforms.unitVec(encoder.encode(span));
  }

  /**
   * @return {0...1}^K dense vector representation of the given phrases
   */
  @Override
  public INDArray encode(Iterable<? extends Span> spans) {
    return Transforms.unitVec(encoder.encode(spans));
  }

  // --- key-value retrieval methods (implements IVocabulary) ---------------------------------------------------------------

  /**
   * @return the {0...1}^K dense vector embedding for a given key.
   */
  @Override
  public INDArray lookup(String key) {
    if(key == null) return null;
    INDArray result = lookupVectors.vector(keyPreprocessor.preProcess(key));
    return result != null ? result.transpose() : null;
  }

  /**
   * @return index of given key, -1 if not found
   */
  @Override
  public int index(String key) {
    if(key == null) return -1;
    final int idx = keyVocabulary.indexOf(keyPreprocessor.preProcess(key));
    return idx >= 0 ? idx : -1; // AbstractCache returns -2 for unknown word
  }

  /**
   * @return key for given index
   */
  @Override
  public String key(int index) {
    VocabWord w = keyVocabulary.elementAtIndex(index);
    return w != null ? w.getWord() : null;
  }

  public List<String> keys() {
    List<String> result = new ArrayList<>(size());
    for(int i = 0; i < size(); i++) {
      result.add(keyVocabulary.wordAtIndex(i));
    }
    return result;
  }
  
  /**
   * @return size of the keyVocabulary N
   */
  @Override
  public int size() {
    return keyVocabulary.numWords();
  }
  
  public long totalInstances() {
    return keyVocabulary.totalWordOccurrences();
  }
  
  /**
   * @return the frequency of a given key in the training corpus
   */
  public double frequency(String key) {
    return keyVocabulary.wordFrequency(keyPreprocessor.preProcess(key));
  }

  /**
   * @return the frequency of a given key in the training corpus
   */
  public double frequency(int index) {
    final String word = key(index);
    return word != null ? keyVocabulary.wordFrequency(word) : 0.;
  }

  /**
   * @return the probability of seeing the key in the corpus (frequency / size)
   */
  public double probability(String key) {
    return frequency(key) / totalInstances();
  }

  /**
   * @return the probability of seeing the key in the corpus (frequency / size)
   */
  public double probability(int index) {
    return frequency(index) / totalInstances();
  }

  // --- knn retrieval methods(implements IVectorIndex) ----------------------------------------------------------------

  /**
   * @return sparse array {0,1}^N for given key
   */
  public INDArray decode(String key) {
    int index = index(key);
    Validate.isTrue(index >= 0, "key is not contained in index");
    return decode(index);
  }

  /**
   * @return sparse array {0,1}^N for given index
   */
  public INDArray decode(int index) {
    Validate.isTrue(index >= 0 && index < size(), "index out of bounds");
    return Nd4j.zeros(DataType.FLOAT, size(), 1).putScalarUnsafe(index, 1.);
  }
  
  /**
   * @return similarity array {0...1}^N of all keys with {0...1}^K dense query vector
   */
  public INDArray similarity(INDArray vec) {
    Validate.isTrue(vec.isColumnVector(), "column vector expected");
    Validate.isTrue(vec.length() == getEmbeddingVectorSize(), "invalid vector size");
    INDArray syn0 = lookupVectors.getSyn0();
    INDArray query = vec.transpose();
    return Transforms.unitVec(query).mmul(syn0.transpose()).transpose();
  }
  
  @Override
  public List<IndexEntry> find(INDArray vec, int k) {
    INDArray sim = similarity(vec);
    List<Double> highToLowSimList = getTopN(sim, k);
    List<IndexEntry> result = new ArrayList<>(k);
    for(int i = 0; i < highToLowSimList.size(); i++) {
      IndexEntry entry = new IndexEntry();
      entry.index = highToLowSimList.get(i).intValue();
      entry.key = key(entry.index);
      entry.similarity = sim.getDouble(entry.index);
      if(entry.similarity != 0.00) result.add(entry); // skip entries with exactly 0 similarity (e.g. NaN)
    }
    return result;
  }
  
  /**
   * Get top N elements
   *
   * @param vec the vec to extract the top elements from
   * @param N the number of elements to extract
   * @return the indices and the sorted top N elements
   */
  private List<Double> getTopN(INDArray vec, int N) {
    BasicModelUtils.ArrayComparator comparator = new BasicModelUtils.ArrayComparator();
    PriorityQueue<Double[]> queue = new PriorityQueue<>(vec.rows(), comparator);
    
    for (int j = 0; j < vec.length(); j++) {
      final Double[] pair = new Double[] {vec.getDouble(j), (double) j};
      if (queue.size() < N) {
        queue.add(pair);
      } else {
        Double[] head = queue.peek();
        if (comparator.compare(pair, head) > 0) {
          queue.poll();
          queue.add(pair);
        }
      }
    }
    
    List<Double> lowToHighSimLst = new ArrayList<>();
    
    while (!queue.isEmpty()) {
      double ind = queue.poll()[1];
      lowToHighSimLst.add(ind);
    }
    return Lists.reverse(lowToHighSimLst);
  }

  @Override
  public IndexEntry find(INDArray vec) {
    // TODO: list might be zero. Should we use Optional?
    return find(vec, 1).get(0);
  }
  
  public void writeVectors(Resource path, String name) throws IOException {
    writeVectors(path, name, null);
  }
  
  /**
   * Write out stored vectors and metadata for use in GloVe or Embedding Projector.
   */
  public void writeVectors(Resource path, String name, Map<String, String> metaMapping) throws IOException {
    
    Resource vecFile = path.resolve(name + ".vectors.tsv");
    Resource metaFile = path.resolve(name + ".meta.tsv");
    Resource gloveFile = path.resolve(name + ".glove.txt");
    
    PrintWriter vecWriter = new PrintWriter(new OutputStreamWriter(vecFile.getOutputStream(), StandardCharsets.UTF_8));
    PrintWriter metaWriter = new PrintWriter(new OutputStreamWriter(metaFile.getOutputStream(), StandardCharsets.UTF_8));
    PrintWriter gloveWriter = new PrintWriter(new OutputStreamWriter(gloveFile.getOutputStream(), StandardCharsets.UTF_8));
    
    try {
      
      int numWords = keyVocabulary.numWords();
      StringBuilder gloveBuilder;
      StringBuilder vecBuilder;
      StringBuilder metaBuilder;
      
      metaWriter.println("Key\tFreq");
      
      for(int i = 0; i < numWords; i++) {
        
        VocabWord word = keyVocabulary.elementAtIndex(i);
        String key = word.getLabel();
        INDArray vec = lookupVectors.vector(word.getLabel());
        gloveBuilder = new StringBuilder();
        vecBuilder = new StringBuilder();
        metaBuilder = new StringBuilder();
        String mappedKey = metaMapping != null ? metaMapping.getOrDefault(key, key) : key;
        metaBuilder.append(mappedKey).append("\t").append((int) word.getElementFrequency());
        gloveBuilder.append(key.replaceAll("\\s+", "_")).append(" ");
        
        for(int k = 0; k < vec.length(); k++) {
          String val = fDbl8(vec.getDouble(k));
          gloveBuilder.append(val);
          vecBuilder.append(val);
          if(k < vec.length() - 1) {
            gloveBuilder.append(" ");
            vecBuilder.append("\t");
          }
        }
        
        // write vectors, metadata and glove files
        vecWriter.println(vecBuilder.toString());
        metaWriter.println(metaBuilder.toString());
        gloveWriter.println(gloveBuilder.toString());
        
      }
      
    } finally {
      vecWriter.flush();
      metaWriter.flush();
      gloveWriter.flush();
      vecWriter.close();
      metaWriter.close();
      gloveWriter.close();
    }
    
  }
  
  protected static String fDbl8(double d) {
    return String.format(Locale.ENGLISH, "%.8f", d);
  }

}
