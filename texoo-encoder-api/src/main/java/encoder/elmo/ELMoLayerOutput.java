package encoder.elmo;

public enum ELMoLayerOutput {
  TOP("top"),
  MIDDLE("middle"),
  BOTTOM("bottom"),
  AVERAGE("average");

  private String path;

  ELMoLayerOutput(String path) {
    this.path = path;
  }

  public String getPath() {
    return path;
  }
}
