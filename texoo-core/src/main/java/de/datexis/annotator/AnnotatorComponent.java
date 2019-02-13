package de.datexis.annotator;

import de.datexis.encoder.IEncoder;
import de.datexis.tagger.Tagger;
import de.datexis.common.Resource;
import de.datexis.common.Timer;
import de.datexis.encoder.Encoder;
import de.datexis.encoder.EncoderSet;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.StreamSupport;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.shade.jackson.annotation.JsonIgnore; // it is import to use the nd4j version in this class!
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;
import org.nd4j.shade.jackson.annotation.JsonInclude.Include;
import org.nd4j.shade.jackson.core.JsonProcessingException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Superclass for Components in an Annotator
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonInclude(Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public abstract class AnnotatorComponent implements IComponent {
    
  protected Logger log = LoggerFactory.getLogger(AnnotatorComponent.class);

  // --- component parameters -------------------------------------------------

  protected String id;
  protected String name;
  protected String model = null;

  protected Timer timer = new Timer();

  private StringBuilder trainLog = new StringBuilder();
  private StringBuilder testLog = new StringBuilder();

  protected boolean modelAvailable = false;

  // --- constructors ----------------------------------------------------------

  public AnnotatorComponent(boolean modelAvailable) {
    this.modelAvailable = modelAvailable;
  }

  // --- property getters / setters --------------------------------------------

  @Override
  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  @Override
  public String getId() {
    return id;
  }

  public void setId(String id) {
    this.id = id;
  }
  
  public void setModelAvailable(boolean modelAvailable) {
    this.modelAvailable = modelAvailable;
  }
  
  /**
   * @return True, iff all models in this Component (including children) are loaded and trained.
   */
  @Override
  @JsonIgnore
  public boolean isModelAvailable() {
    return modelAvailable && isModelAvailableInChildren();
  }
  
  /**
   * @return True, iff all models in all children components are loaded and trained.
   */
  @JsonIgnore
  public boolean isModelAvailableInChildren() {
    return true;
  }
  
  /**
   * @return JSON representation of Component configuration
   */
  @JsonIgnore
  public String getConf() {
    try {
      String json = NeuralNetConfiguration.mapper().writer().writeValueAsString(this);
      return json.replaceAll("\\s", "");
    } catch (JsonProcessingException ex) {
      log.error("Could not serialize class to JSON: " + ex.toString());
      return null;
    }
  }

  /**
   * Initializes the Component and sets configuration
   */
  public void setConf() {
    throw new UnsupportedOperationException();
  }

  // --- serialization getters / setters ---------------------------------------

  /**
   * @return Model reference as String (file reference or URL)
   */
  @JsonIgnore
  public String getModel() {
    return model == null ? "" : model;
  }

  /**
   * Sets the model Resource. Called by loadModel or saveModel.
   * @param model 
   */
  protected void setModel(Resource model) {
    if(model == null) this.model = null;
    else this.model = model.getFileName();
  }
  
  protected void setModelFilename(String model) {
    this.model = model;
  }

  public void appendTrainLog(String message) {
    trainLog.append(message).append("\n");
    log.info(message);
  }
  
  public void appendTrainLog(String message, long time) {
    String msg = message + " [" + Timer.millisToLongDHMS(time) + "]";
    trainLog.append(msg).append("\n");
    log.info(msg);
  }
  
  public void appendTestLog(String message) {
    testLog.append(message).append("\n");
    //log.info(message);
  }
  
  public void appendTestLog(String message, long time) {
    String msg = message + " [" + Timer.millisToLongDHMS(time) + "]";
    testLog.append(msg).append("\n");
    log.info(msg);
  }
  
  protected String getTrainLog() {
    return trainLog.toString();
  }
  
  protected String getTestLog() {
    return testLog.toString();
  }

  protected void clearTrainLog() {
    trainLog = new StringBuilder();
  }
  
  protected void clearTestLog() {
    testLog = new StringBuilder();
  }
 
}