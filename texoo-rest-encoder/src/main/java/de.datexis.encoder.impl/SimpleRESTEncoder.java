package de.datexis.encoder.impl;

import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;

public abstract class SimpleRESTEncoder extends AbstractRESTEncoder {
  private static final Logger log = LoggerFactory.getLogger(SimpleRESTEncoder.class);

  private Class<? extends Span> elementClass;

  public SimpleRESTEncoder(RESTAdapter restAdapter, Class<? extends Span> elementClass) {
    super(restAdapter);
    this.elementClass = elementClass;
  }

  public SimpleRESTEncoder(RESTAdapter restAdapter, String vectorIdentifier, Class<? extends Span> elementClass) {
    super(restAdapter, vectorIdentifier);
    this.elementClass = elementClass;
  }

  @Override
  public INDArray encode(String word) {
    try{
      return encodeImpl(word);
    }catch (IOException e){
      log.error("IO Error while encoding word: {}", word, e);
      throw new UncheckedIOException(e);
    }
  }

  public abstract INDArray encodeImpl(String word) throws IOException;

  @Override
  public INDArray encode(Span span) {
    if(elementClass.isInstance(span)){
      try{
        return encodeImpl(span);
      }catch (IOException e){
        log.error("IO Error while encoding span: {}", span, e);
        throw new UncheckedIOException(e);
      }
    }else{
      throw new UnsupportedOperationException();
    }
  }

  public abstract INDArray encodeImpl(Span span) throws IOException;

  @Override
  public void encodeEach(Sentence input, Class<? extends Span> elementClass) {
    if(elementClass == this.elementClass){
      try {
        encodeEachImpl(input);
      }catch (IOException e){
        log.error("IO Error while encoding sentence: {}", input, e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException();
    }
  }

  public abstract void encodeEachImpl(Sentence input) throws IOException;

  @Override
  public void encodeEach(Document input, Class<? extends Span> elementClass) {
    if(elementClass == this.elementClass){
      try {
        encodeEachImpl(input);
      }catch (IOException e){
        log.error("IO Error while encoding document: {}", input.getTitle(), e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException();
    }
  }

  public abstract void encodeEachImpl(Document input) throws IOException;

  @Override
  public void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass) {
    if(elementClass == this.elementClass){
      try {
        encodeEachImpl(docs);
      }catch (IOException e){
        log.error("IO Error while encoding documents", e);
        throw new UncheckedIOException(e);
      }
    } else {
      throw new UnsupportedOperationException();
    }
  }

  public abstract void encodeEachImpl(Collection<Document> docs) throws IOException;
}
