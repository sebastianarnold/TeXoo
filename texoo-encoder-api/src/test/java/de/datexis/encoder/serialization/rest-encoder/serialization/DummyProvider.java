package de.datexis.encoder.impl.serde;

import encoder.serialization.DeserializationProvider;
import encoder.serialization.SerializationProvider;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class DummyProvider implements SerializationProvider, DeserializationProvider {
  public static final String ACCEPT_TYPE = "application/accept";
  public static final String CONTENT_TYPE = "application/content";

  @Override
  public <O> O deserialize(InputStream inputStream, Class<O> classOfO) throws IOException {
    return null;
  }

  @Override
  public String getAcceptType() {
    return ACCEPT_TYPE;
  }

  @Override
  public <I> void serialize(I input, OutputStream outputStream) throws IOException {

  }

  @Override
  public String getContentType() {
    return CONTENT_TYPE;
  }
}
