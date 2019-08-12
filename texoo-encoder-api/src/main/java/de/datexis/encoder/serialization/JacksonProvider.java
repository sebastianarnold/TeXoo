package de.datexis.encoder.serialization;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class JacksonProvider implements SerializationProvider, DeserializationProvider {
  private ObjectMapper objectMapper = new ObjectMapper();

  @Override
  public <O> O deserialize(InputStream inputStream, Class<O> classOfO) throws IOException {
    return objectMapper.readValue(inputStream, classOfO);
  }

  @Override
  public <I> void serialize(I input, OutputStream outputStream) throws IOException {
    objectMapper.writeValue(outputStream, input);
  }

  @Override
  public String getContentType() {
    return "application/json; charset=UTF-8";
  }

  @Override
  public String getAcceptType() {
    return "application/json; charset=UTF-8";
  }
}
