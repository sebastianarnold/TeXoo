package de.datexis.encoder.serialization;

import java.io.IOException;
import java.io.OutputStream;

public interface SerializationProvider {
  <I> void serialize(I input, OutputStream outputStream) throws IOException;

  String getContentType();
}
