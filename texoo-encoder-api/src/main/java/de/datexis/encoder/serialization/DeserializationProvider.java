package de.datexis.encoder.serialization;

import java.io.IOException;
import java.io.InputStream;

public interface DeserializationProvider {
  <O> O deserialize(InputStream inputStream, Class<O> classOfO) throws IOException;

  String getAcceptType();
}
