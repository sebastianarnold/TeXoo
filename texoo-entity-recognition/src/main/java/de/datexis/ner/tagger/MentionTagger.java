package de.datexis.ner.tagger;

import com.google.common.collect.Lists;
import de.datexis.encoder.EncoderSet;
import de.datexis.model.*;
import de.datexis.model.tag.*;
import static de.datexis.model.tag.Tag.GENERIC;
import de.datexis.ner.MentionAnnotation;
import de.datexis.ner.eval.MentionAnnotatorEval;
import de.datexis.ner.eval.MentionTaggerEval;
import de.datexis.tagger.AbstractIterator;
import de.datexis.tagger.Tagger;
import java.util.ArrayList;
import java.util.Collection;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional.Mode;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Assigns BIO2 or BIOES Labels to every Token in a Document.
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
public class MentionTagger extends Tagger {

  protected static final Logger log = LoggerFactory.getLogger(MentionTagger.class);
  
  /**
   * Number of training examples
   */
  private int n = 0;
  
  protected int batchSize = 32;
  protected int numEpochs = 1;
  protected boolean randomize = true;
  protected int workers = 4;
  
  protected Class<? extends Tag> tagset = BIOESTag.class;
  protected String types = Tag.GENERIC;
  
  protected MentionTaggerEval eval;
  
  public MentionTagger() {
    this("BLSTM");
    setTagset(BIOESTag.class, Tag.GENERIC);
  }
  
  public MentionTagger(String id) {
    super(id);
    setTagset(BIOESTag.class, Tag.GENERIC);
    eval = new MentionTaggerEval("loaded");
  }
  
  public MentionTagger(AbstractIterator data, int ffwLayerSize, int lstmLayerSize, int iterations, double learningRate) {
    super(data.getInputSize(), data.getLabelSize());
		net = createBLSTM(inputVectorSize, ffwLayerSize, lstmLayerSize, outputVectorSize, iterations, learningRate);
	}
  
  public MentionTagger build(int ffwLayerSize, int lstmLayerSize, int iterations, double learningRate) {
    net = createBLSTM(inputVectorSize, ffwLayerSize, lstmLayerSize, outputVectorSize, iterations, learningRate);
    return this;
  }
  
  public Class<? extends Tag> getTagset() {
    return tagset;
  }
  
  public static ComputationGraph createBLSTM(long inputVectorSize, long ffwLayerSize, long lstmLayerSize, long outputVectorSize, int iterations, double learningRate) {

		log.info("initializing BLSTM network " + inputVectorSize + ":" + ffwLayerSize + ":" + ffwLayerSize + ":" + lstmLayerSize + ":" + outputVectorSize);
		ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				//.updater(new RmsProp(0.95))
        .updater(new Adam(learningRate, 0.9, 0.999, 1e-8))
        .l2(0.0001)
        .trainingWorkspaceMode(WorkspaceMode.ENABLED)
        .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.graphBuilder()
        .addInputs("input");
      if(ffwLayerSize > 0) {
        gb.addLayer("FF1", new DenseLayer.Builder()
            .nIn(inputVectorSize).nOut(ffwLayerSize)
            .activation(Activation.RELU)
            .weightInit(WeightInit.RELU)
            .build(), /*new RnnToFeedForwardPreProcessor(),*/ "input")
        .addLayer("FF2", new DenseLayer.Builder()
            .nIn(ffwLayerSize).nOut(ffwLayerSize)
            .activation(Activation.RELU)
            .weightInit(WeightInit.RELU)
            .build(), "FF1")
        .addLayer("BLSTM", new Bidirectional(Mode.AVERAGE, new LSTM.Builder()
            .nIn(ffwLayerSize).nOut(lstmLayerSize)
						.activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            //.dropOut(0.5)
            .build()), "FF2");
      } else {
        gb.addLayer("BLSTM", new Bidirectional(Mode.AVERAGE, new LSTM.Builder()
            .nIn(inputVectorSize).nOut(lstmLayerSize)
						.activation(Activation.TANH)
            .weightInit(WeightInit.XAVIER)
            //.dropOut(0.5)
            .build()), "input");
      }
        gb.addLayer("output", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
            .nIn(lstmLayerSize).nOut(outputVectorSize)
            .activation(Activation.SOFTMAX)
            .weightInit(WeightInit.XAVIER)
            .build(), "BLSTM")
        .setOutputs("output")
        .setInputTypes(InputType.recurrent(inputVectorSize))
				.pretrain(false).backprop(true).backpropType(BackpropType.Standard)
        .build();

    ComputationGraphConfiguration conf = gb.build();
		ComputationGraph lstm = new ComputationGraph(conf);
		lstm.init();
		return lstm;
    
  }
  
  /**
	 * creates a GravesLSTM with given configuration
	 * 
	 * @param inputVectorSize
   * @param ffwLayerSize
	 * @param lstmLayerSize
	 * @param outputVectorSize
   * @param iterations
   * @param learningRate
	 * @return
	 */
  @Deprecated
	public static MultiLayerNetwork createBLSTMNet(int inputVectorSize, int ffwLayerSize, int lstmLayerSize, int outputVectorSize, int iterations, double learningRate) {

		log.info("initializing FF+BLSTM network " + inputVectorSize + ":" + ffwLayerSize + ":" + ffwLayerSize + ":" + lstmLayerSize + ":" + outputVectorSize);
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .l2(0.0001)
        .weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        //.updater(Updater.RMSPROP).rmsDecay(0.95)
        //.updater(Updater.NESTEROVS).momentum(0.9) // momentum
        .updater(new Adam(learningRate, 0.9, 0.999, 1e-8))
				.list()
        .layer(0, new DenseLayer.Builder().nIn(inputVectorSize).nOut(ffwLayerSize)
                .activation(Activation.RELU)
                .weightInit(WeightInit.RELU)
                .build())
        .layer(1, new DenseLayer.Builder().nIn(ffwLayerSize).nOut(ffwLayerSize)
                .activation(Activation.RELU)
                .weightInit(WeightInit.RELU)
                .build())
        .layer(2, new GravesBidirectionalLSTM.Builder().nIn(ffwLayerSize).nOut(lstmLayerSize)
								.activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                //.dropOut(0.5)
                .build())
        .layer(3, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).nIn(lstmLayerSize).nOut(outputVectorSize)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
				.inputPreProcessor(0, new RnnToFeedForwardPreProcessor())
        .inputPreProcessor(2, new FeedForwardToRnnPreProcessor())
				.setInputType(InputType.recurrent(inputVectorSize))
				.backprop(true).pretrain(false)
        .build();

		MultiLayerNetwork lstm = new MultiLayerNetwork(conf);
		lstm.init();
    lstm.setLearningRate(learningRate);
		return lstm;
  }
  
  public MentionTagger setTagset(Class<? extends Tag> tagset) {
    this.tagset = tagset;
    try {
      this.outputVectorSize = tagset.newInstance().getVectorSize();
    } catch (Exception ex) {
      log.error("Could not set output vector size");
    }
    return this;
  }
  
  public MentionTagger setTagset(Class<? extends Tag> tagset, String types) {
    setTagset(tagset);
    this.types = types;
    return this;
  }
  
  @Override
  public MentionTagger setEncoders(EncoderSet encoderSet) {
    super.setEncoders(encoderSet);
    this.inputVectorSize = encoderSet.getEmbeddingVectorSize();
    return this;
  }
  
  public MentionTagger setTrainingParams(int batchSize, int numEpochs, boolean randomize) {
    this.batchSize = batchSize;
    this.numEpochs = numEpochs;
    this.randomize = randomize;
    return this;
  }
  
  public MentionTagger setWorkspaceParams(int workers) {
    this.workers = workers;
    return this;
  }
  
  public void activateCUDA() {
    //DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
     //CudaEnvironment.getInstance().getConfiguration()
            // key option enabled
//            .allowMultiGPU(true)
            // we're allowing larger memory caches
  //          .setMaximumDeviceCache(2L * 1024L * 1024L * 1024L)
            // cross-device access is used for faster model averaging over pcie
    //        .allowCrossDeviceAccess(true);
  }
  
  @Override
  public void trainModel(Dataset dataset, Annotation.Source trainingAnnotations) {
    trainModel(dataset, trainingAnnotations, -1, randomize);
  }
  
  public void trainModel(Dataset dataset, Annotation.Source trainingAnnotations, int numExamples, boolean randomize) {
    trainModel(new MentionTaggerIterator(dataset.getDocuments(), dataset.getName(), encoders, tagset, trainingAnnotations, numExamples, batchSize, randomize));
  }
  
  public void trainModel(Collection<Sentence> sentences, Annotation.Source trainingTags, boolean randomize) {
    trainModel(new MentionTaggerIterator(Lists.newArrayList(new Sample(sentences, randomize)), "training", encoders, tagset, trainingTags, -1, batchSize, randomize));
  }
  
  protected void trainModel(MentionTaggerIterator it) {
    int batches = it.numExamples() / it.batch();
    int n = 0;
    
    appendTrainLog("Training " + getName() + " with " + it.numExamples() + " examples in " + batches + " batches for " + numEpochs + " epochs.");
    
    // ParallelWrapper will take care of load balancing between GPUs.
    ParallelWrapper wrapper = null;
    if(workers > 1) {
      wrapper = new ParallelWrapper.Builder(net)
        .prefetchBuffer(workers * 4)  // DataSets prefetching options. Set this value with respect to number of actual devices
        .workers(workers)          // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
        //.averagingFrequency(1) // rare averaging improves performance, but might reduce model accuracy
        //.reportScoreAfterAveraging(false) // if set to TRUE, on every averaging model score will be reported
        .trainingMode(ParallelWrapper.TrainingMode.SHARED_GRADIENTS)
        .workspaceMode(WorkspaceMode.SINGLE)
        .build();
    }
    eval = new MentionTaggerEval(getName(), tagset);
    timer.start();
		for(int i = 1; i <= numEpochs; i++) {
      timer.setSplit("epoch");
      if(wrapper != null) {
        wrapper.fit(it);
      } else {
        if(net instanceof ComputationGraph) ((ComputationGraph) net).fit(it);
        else if(net instanceof MultiLayerNetwork) ((MultiLayerNetwork) net).fit(it);
      }
      n += it.numExamples();
			appendTrainLog("Completed epoch " + i + " of " + numEpochs + "\t" + n, timer.getLong("epoch"));
      it.reset();
		}
    timer.stop();
		appendTrainLog("Training complete", timer.getLong());
    eval.setTrainDataset(new Dataset(), n, timer.getLong());
    setModelAvailable(true);
    
  }
  
  /**
   * Predicts labels for all Tokens in the Iterator and assigns Tags (BIO2 or BIOES).
   * requires: Encoder.class on Token.class (using parallelized DocumentIterator batches)
   * attaches: BIO2Tag.class to Token.class
   */
  @Override
  public synchronized void tag(Collection<Document> documents) {
    log.debug("Labeling Documents...");
    MentionTaggerIterator it = new MentionTaggerIterator(documents, "train", encoders, tagset, -1, batchSize, false);
    it.reset();
		while(it.hasNext()) {
      // 1. Load a batch of sentences
      Pair<DataSet,ArrayList<Sentence>> examples = it.nextDataSet();
			INDArray input = examples.getKey().getFeatures();
      INDArray inputMask = examples.getKey().getFeaturesMaskArray();
      INDArray labelsMask = examples.getKey().getLabelsMaskArray();
      // 2. Predict labels
      INDArray predicted = null;
      if(net instanceof MultiLayerNetwork) {
        predicted = ((MultiLayerNetwork) net).output(input, false, inputMask, labelsMask);
      } else if(net instanceof ComputationGraph) {
        ((ComputationGraph) net).setLayerMaskArrays(new INDArray[]{inputMask}, new INDArray[]{labelsMask});
        predicted = ((ComputationGraph) net).outputSingle(input);
      }
      // 3. Create BIOES tags from vectors + CRF and convert to BIO2 - RENAME
      createTags(examples.getValue(), predicted, it.getTagset(), Annotation.Source.PRED, false, true);
		}
    for(Document doc : it.getDocuments()) {
      doc.setTagAvailable(Annotation.Source.PRED, it.getTagset(), true);
      if(!tagset.equals(BIO2Tag.class)) doc.setTagAvailable(Annotation.Source.PRED, BIO2Tag.class, true);
    }
  }
  
  public void tagSentences(Collection<Sentence> sentences) {
    tag(Lists.newArrayList(new Sample(sentences, false)));
  }
    
  /**
   * requires: GOLD BIO2Tag.class and BIOESTag.class for Token.class
   * attaches: PRED BIO2Tag.class and BIOESTag.class to Token.class
   * @param dataset
   * @param expected 
   */
  @Override
  public void testModel(Dataset dataset, Annotation.Source expected) {
    
    // Tag Dataset using BIOES and finally produce BIO2 tags
    MentionTaggerIterator it = new MentionTaggerIterator(dataset.getDocuments(), dataset.getName(), encoders, tagset, -1, batchSize, false);
    test(it);
    
    // Test tagging performance: BIOES or BIO2
    MentionTaggerEval eval = new MentionTaggerEval(getName(), tagset);
    eval.calculateMeasures(dataset);
    appendTestLog(eval.printExperimentStats());
    appendTestLog(eval.printDatasetStats());
    appendTestLog(eval.printTrainingCurve());
    appendTestLog(eval.printSequenceClassStats(false));
    
    // Test annotation performance: exact match using BIO2 tags
    MentionAnnotatorEval annE = new MentionAnnotatorEval(getName());
    for(Document doc : dataset.getDocuments()) {
      if(doc.countAnnotations(expected) == 0)
        MentionAnnotation.annotateFromTags(doc, expected, BIO2Tag.class, types);
      doc.clearAnnotations(Annotation.Source.PRED, MentionAnnotation.class);
      MentionAnnotation.annotateFromTags(doc, Annotation.Source.PRED, BIO2Tag.class, types);
    }
    annE.setTestDataset(dataset, 0, 0);
    annE.evaluateAnnotations();
    appendTestLog(annE.printAnnotationStats());
    
  }
  
  public Evaluation test(MentionTaggerIterator it) {
    timer.start();
    appendTrainLog("Evaluating " + getName() + " with " + it.numExamples() + " examples...");
    Evaluation eval = new Evaluation(it.getLabelSize());
    it.reset();
    while(it.hasNext()) {
      // 1. Load a batch of sentences
      Pair<DataSet,ArrayList<Sentence>> examples = it.nextDataSet();
			INDArray input = examples.getKey().getFeatures();
      INDArray labels = examples.getKey().getLabels();
      INDArray inputMask = examples.getKey().getFeaturesMaskArray();
      INDArray labelsMask = examples.getKey().getLabelsMaskArray();
      // 2. Predict labels
			INDArray predicted = null;
      if(net instanceof MultiLayerNetwork) {
        predicted = ((MultiLayerNetwork) net).output(input, false, inputMask, labelsMask);
      } else if(net instanceof ComputationGraph) {
        ((ComputationGraph) net).setLayerMaskArrays(new INDArray[]{inputMask}, new INDArray[]{labelsMask});
        predicted = ((ComputationGraph) net).outputSingle(input);
      }
      try {
        eval.evalTimeSeries(labels, predicted, labelsMask);
      } catch(IllegalStateException ex) {
        log.warn(ex.toString());
      }
      // 3. Create tags from labels
      createTags(examples.getValue(), predicted, it.getTagset(), Annotation.Source.PRED, true, true);
    }
    for(Document doc : it.getDocuments()) {
      doc.setTagAvailable(Annotation.Source.PRED, it.getTagset(), true);
      if(!tagset.equals(BIO2Tag.class)) doc.setTagAvailable(Annotation.Source.PRED, BIO2Tag.class, true);
    }
    timer.stop();
		appendTrainLog("Evaluation complete", timer.getLong());
    return eval;
  }
  
  public void enableTrainingUI() {
    StatsStorage stats = new InMemoryStatsStorage();
    net.addListeners(new StatsListener(stats, 1));
    UIServer.getInstance().attach(stats);
  }
  
  /**
   * Creates BIO2Tags from predictions.
   * requires: INDArray predictions for Token.class
   * attaches: BIO2Tag.class to Token.class
   * @param sents List of sentences to tag.
   * @param predicted Predictions from the sequence model.
   * @param tagset The tagset to use for tagging (e.g. BIOES). Output will be transformed to BIOES then.
   * @param source Which tags to use: GOLD, PREDICTED or USER.
   * @param keepVectors If TRUE, vectors are not deleted from the Tokens.
   * @param convertTags If TRUE, tags are corrected and converted to BIO2. Otherwise, we keep Tags of class tagset.
   */
  public static void createTags(Iterable<Sentence> sents, INDArray predicted, Class tagset, Annotation.Source source, boolean keepVectors, boolean convertTags) {
    //System.out.println(predicted.toString());
    int batchNum = 0, t = 0;
    for(Sentence s : sents) {
      for(Token token : s.getTokens()) {
        INDArray vec = predicted.getRow(batchNum).getColumn(t++);
        // FIXME: we cannot simply assign GENERIC here!
        if(tagset.equals(BIO2Tag.class)) token.putTag(source, new BIO2Tag(vec, GENERIC, true));
        if(tagset.equals(BIOESTag.class)) token.putTag(source, new BIOESTag(vec, GENERIC, true));
      }
      t=0; batchNum++;
      if(tagset.equals(BIOESTag.class)) {
        BIOESTag.correctCRF(s, source);
        if(convertTags) BIOESTag.convertToBIO2(s, source);
      }
      // TODO: cleanup the vectors here now!
      if(!keepVectors) {
        
      }
    }
  }
  
}
