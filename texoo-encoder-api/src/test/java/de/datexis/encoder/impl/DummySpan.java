package de.datexis.encoder.impl;

import de.datexis.model.Span;

public class DummySpan extends Span {
  public static final String TEXT = "text";
  public static final String CUSTOM_TEXT = "custom text";

  public DummySpan() {
  }

  @Override
  public String getText() {
    return TEXT;
  }

  public String customGetText(){
    return CUSTOM_TEXT;
  }
}
