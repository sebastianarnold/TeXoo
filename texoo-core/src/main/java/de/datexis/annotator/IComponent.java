package de.datexis.annotator;

import de.datexis.common.Resource;
import de.datexis.encoder.IEncoder;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonInclude;

import java.io.IOException;

/**
 * Superclass for Components in an Annotator
 * @author Sebastian Arnold <sarnold@beuth-hochschule.de>
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonIgnoreProperties(ignoreUnknown = true)
public interface IComponent {

  String getName();

  String getId();

  @JsonIgnore
  boolean isModelAvailable();

  /**
	 * Load a pre-trained model
   * @param file The file to load
	 */
  void loadModel(Resource file) throws IOException;

  /**
	 * Load a pre-trained model
   * @param dir The path to create the file
   * @param name The name of the model. File extension will be added automatically.
	 */
  void saveModel(Resource dir, String name);

}
