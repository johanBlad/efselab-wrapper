echo "BUILD LEMMATIZER"
python setup.py build_ext --inplace

# Download and build models
echo "DOWNLOAD AND BUILD TAGGER AND NER MODELS"
if [ ! -e swe-pipeline-ud2.tar.gz ]; then
    wget http://mumin.ling.su.se/projects/efselab/swe-pipeline-ud2.tar.gz
    tar xvzf swe-pipeline-ud2.tar.gz
    python build_suc.py --skip-generate --python --n-train-fields 2
    python build_suc_ne.py --skip-generate --python --n-train-fields 4
fi


# Build and train the SUC-to-UD conversion model
echo "BUILD AND TRAIN SUC-UD CONVERSION MODEL"
python build_udt_suc_sv.py --python --beam-size 1 --n-train-fields 4
./udt_suc_sv train \
    data/sv-ud-train.tab data/sv-ud-dev.tab swe-pipeline/suc-ud.bin

# Copy necessary binaries to bin folder, for usage in the pipeline