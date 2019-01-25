#!/bin/bash
set -e

# Copy from glove repository, here embedding size is 300

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python


DATA_DIR=/data/research/data/textMatch/
OUTPUT_DIR=/data/research/data/textMatch/output/

CORPUS=$DATA_DIR/content.txt
VOCAB_FILE=$DATA_DIR/vocab_word.txt
COOCCURRENCE_FILE=$OUTPUT_DIR/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=$OUTPUT_DIR/cooccurrence.shuf.bin
BUILDDIR=/data/research/github/glove/
SAVE_FILE=$OUTPUT_DIR/vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=300
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10

echo
echo "$ $BUILDDIR/build/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/build/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
echo "$ $BUILDDIR/build/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/build/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
echo "$ $BUILDDIR/build/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/build/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
echo "$ $BUILDDIR/build/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/build/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

#echo "$ python $BUILDDIR/eval/python/evaluate.py"
#python $BUILDDIR/eval/python/evaluate.py --vocab_file $VOCAB_FILE --vectors_file $SAVE_FILE.txt

# convert glove embedding file to numpy format
python merge_embedding.py --vocab_file $DATA_DIR/vocab_word.txt \
    --embedding_file $OUTPUT_DIR/vectors.txt    \
    --output_npy   $DATA_DIR/embedding.npy  \
    --output_vocab  $DATA_DIR/words.txt    \
    --min_count 5  \
    --embedding_dim 300
