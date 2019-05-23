package de.datexis.encoder.impl;

import de.datexis.encoder.impl.AbstractRESTEncoder;
import de.datexis.model.Document;
import de.datexis.model.Sentence;
import de.datexis.model.Span;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.List;

public class DummyAbstractRESTEncoder extends AbstractRESTEncoder {
    @Override
    public long getEmbeddingVectorSize() {
        return 100;
    }

    @Override
    public INDArray encode(String word) {
        return null;
    }

    @Override
    public INDArray encode(Span span) {
        return null;
    }

    @Override
    public INDArray encode(Iterable<? extends Span> spans) {
        return null;
    }

    @Override
    public void encodeEach(Sentence input, Class<? extends Span> elementClass) {

    }

    @Override
    public void encodeEach(Document input, Class<? extends Span> elementClass) {

    }

    @Override
    public void encodeEach(Collection<Document> docs, Class<? extends Span> elementClass) {

    }

    @Override
    public INDArray encodeMatrix(List<Document> input, int maxTimeSteps, Class<? extends Span> timeStepClass) {
        return null;
    }
}
