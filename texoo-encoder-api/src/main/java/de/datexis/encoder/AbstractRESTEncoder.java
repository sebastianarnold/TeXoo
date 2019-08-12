package de.datexis.encoder;

import de.datexis.common.Resource;
import de.datexis.model.Document;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.IOException;
import java.util.Collection;
import java.util.stream.Stream;

/**
 * Interface for an Encoder that runs behind a REST API (no implementation).
 */
public abstract class AbstractRESTEncoder extends Encoder {

  protected RESTAdapter restAdapter;

  protected AbstractRESTEncoder(String id) {
    super(id);
  }
  
  public AbstractRESTEncoder(String id, RESTAdapter restAdapter) {
    super(id);
    this.restAdapter = restAdapter;
  }
  
  @JsonTypeInfo(use=JsonTypeInfo.Id.CLASS, include=JsonTypeInfo.As.PROPERTY, property="class")
  public RESTAdapter getRestAdapter() {
    return restAdapter;
  }
  
  public void setRestAdapter(RESTAdapter restAdapter) {
    this.restAdapter = restAdapter;
  }
  
  @Override
  public long getEmbeddingVectorSize() {
    return restAdapter.getEmbeddingVectorSize();
  }

  @Override
  public void trainModel(Collection<Document> documents) {
    throw new UnsupportedOperationException("REST Encoders are not trainable");
  }

  @Override
  public void trainModel(Stream<Document> documents) {
    throw new UnsupportedOperationException("REST Encoders are not trainable");
  }

  @Override
  public void loadModel(Resource file) throws IOException {
    throw new UnsupportedOperationException("REST Encoders cant load a model");
  }

  @Override
  public void saveModel(Resource dir, String name) throws IOException {
    throw new UnsupportedOperationException("REST Encoders cant save a model");
  }

}
